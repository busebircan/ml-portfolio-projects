"""
Azure Functions - Serverless Dynamic Pricing Prediction
HTTP-triggered function for on-demand predictions
"""

import logging
import json
import joblib
import pandas as pd
import numpy as np
import azure.functions as func
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_config import AzureDataConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables (loaded at cold start)
model = None
scaler = None
connector = None
feature_names = None


def load_model():
    """Load model artifacts (called at cold start)"""
    global model, scaler, connector, feature_names
    
    if model is None:
        logger.info("Loading model artifacts...")
        
        # Model path in Azure Functions
        model_dir = os.getenv('MODEL_DIR', './models')
        
        model = joblib.load(os.path.join(model_dir, 'pricing_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load feature names
        with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Initialize Azure connector
        connector = AzureDataConnector(use_managed_identity=True)
        
        logger.info(f"Model loaded with {len(feature_names)} features")


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main Azure Function handler
    
    HTTP Trigger: POST /api/predict
    
    Request Body:
    {
        "features": [...],  // Direct feature input
        "query": "...",     // SQL query
        "write_to_db": true // Optional: write predictions to database
    }
    """
    logger.info('Processing pricing prediction request')
    
    try:
        # Load model if not already loaded
        load_model()
        
        # Parse request
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f"Request keys: {list(req_body.keys())}")
        
        # Load data based on input type
        if 'features' in req_body:
            # Direct feature input
            df = pd.DataFrame(req_body['features'])
            logger.info(f"Loaded {len(df)} records from direct input")
        
        elif 'query' in req_body:
            # Query from Azure SQL
            query = req_body['query']
            params = req_body.get('params')
            df = connector.load_data_from_sql(query, params=params)
            logger.info(f"Loaded {len(df)} records from SQL")
        
        elif 'blob_container' in req_body and 'blob_name' in req_body:
            # Load from blob storage
            df = connector.load_from_blob(
                container_name=req_body['blob_container'],
                blob_name=req_body['blob_name'],
                file_type=req_body.get('file_type', 'csv')
            )
            logger.info(f"Loaded {len(df)} records from blob")
        
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": "Must provide 'features', 'query', or 'blob_container'/'blob_name'"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return func.HttpResponse(
                json.dumps({
                    "error": f"Missing required features: {list(missing_features)}"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare features
        X = df[feature_names].fillna(df[feature_names].mean())
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Calculate confidence
        confidence = np.ones(len(predictions)) * 0.85
        
        # Prepare results
        results_df = df.copy()
        results_df['predicted_price'] = predictions
        results_df['confidence'] = confidence
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        
        # Calculate price changes if base price available
        if 'base_price' in df.columns:
            results_df['price_change_pct'] = (
                (results_df['predicted_price'] - results_df['base_price']) / 
                results_df['base_price'] * 100
            )
        
        # Write to database if requested
        if req_body.get('write_to_db', False):
            table_name = req_body.get('output_table', 'price_predictions')
            
            output_columns = [
                'product_id', 'predicted_price', 'confidence', 'prediction_timestamp'
            ]
            
            if 'price_change_pct' in results_df.columns:
                output_columns.append('price_change_pct')
            
            connector.write_predictions_to_sql(
                results_df[output_columns],
                table_name=table_name,
                if_exists='append'
            )
            
            logger.info(f"Wrote {len(results_df)} predictions to {table_name}")
        
        # Prepare response
        response = {
            'status': 'success',
            'predictions': predictions.tolist(),
            'confidence': confidence.tolist(),
            'num_predictions': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'mean_predicted_price': float(np.mean(predictions)),
                'median_predicted_price': float(np.median(predictions)),
                'min_predicted_price': float(np.min(predictions)),
                'max_predicted_price': float(np.max(predictions))
            }
        }
        
        if 'price_change_pct' in results_df.columns:
            response['statistics']['mean_price_change_pct'] = float(
                results_df['price_change_pct'].mean()
            )
        
        logger.info(f"Prediction completed successfully")
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        
        return func.HttpResponse(
            json.dumps({
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )
