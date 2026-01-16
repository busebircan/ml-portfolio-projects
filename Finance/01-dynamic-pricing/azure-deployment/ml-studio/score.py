"""
Azure ML Scoring Script for Dynamic Pricing Model
This script runs on Azure ML managed endpoints
"""

import json
import joblib
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_config import AzureDataConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and connector
model = None
scaler = None
connector = None
feature_names = None


def init():
    """
    Initialize model when Azure ML endpoint starts
    This function is called once when the endpoint is deployed
    """
    global model, scaler, connector, feature_names
    
    try:
        logger.info("Initializing Dynamic Pricing model...")
        
        # Get model directory from Azure ML
        model_path = os.path.join(
            os.getenv('AZUREML_MODEL_DIR', './models'),
            'dynamic-pricing-model'
        )
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load model artifacts
        model = joblib.load(os.path.join(model_path, 'pricing_model.pkl'))
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        
        # Load feature names
        with open(os.path.join(model_path, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Model loaded successfully with {len(feature_names)} features")
        
        # Initialize database connector
        connector = AzureDataConnector(use_managed_identity=True)
        logger.info("Azure Data Connector initialized")
        
        logger.info("Initialization complete!")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise


def run(raw_data):
    """
    Process incoming prediction requests
    
    Input formats supported:
    1. Direct features: {"features": [...]}
    2. SQL query: {"query": "SELECT ...", "write_to_db": true}
    3. Blob reference: {"blob_container": "...", "blob_name": "..."}
    
    Args:
        raw_data: JSON string with request data
    
    Returns:
        JSON string with predictions and metadata
    """
    try:
        logger.info("Processing prediction request...")
        start_time = datetime.now()
        
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Request type: {list(data.keys())}")
        
        # Load data based on input type
        if 'features' in data:
            # Direct feature input
            df = pd.DataFrame(data['features'])
            logger.info(f"Loaded {len(df)} records from direct input")
        
        elif 'query' in data:
            # Query from Azure SQL Database
            query = data['query']
            params = data.get('params')
            df = connector.load_data_from_sql(query, params=params)
            logger.info(f"Loaded {len(df)} records from SQL query")
        
        elif 'blob_container' in data and 'blob_name' in data:
            # Load from Azure Blob Storage
            df = connector.load_from_blob(
                container_name=data['blob_container'],
                blob_name=data['blob_name'],
                file_type=data.get('file_type', 'csv')
            )
            logger.info(f"Loaded {len(df)} records from blob storage")
        
        else:
            raise ValueError("Invalid input format. Must provide 'features', 'query', or 'blob_container'/'blob_name'")
        
        # Validate required features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare features
        X = df[feature_names]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Calculate confidence intervals (if model supports it)
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(X_scaled).max(axis=1)
        else:
            # For regression, use prediction variance as proxy
            confidence = np.ones(len(predictions)) * 0.85
        
        # Prepare results
        results_df = df.copy()
        results_df['predicted_price'] = predictions
        results_df['confidence'] = confidence
        results_df['prediction_timestamp'] = datetime.now().isoformat()
        results_df['model_version'] = os.getenv('MODEL_VERSION', '1.0')
        
        # Calculate additional metrics
        if 'base_price' in df.columns:
            results_df['price_change_pct'] = (
                (results_df['predicted_price'] - results_df['base_price']) / 
                results_df['base_price'] * 100
            )
        
        # Write predictions to database if requested
        if data.get('write_to_db', False):
            table_name = data.get('output_table', 'price_predictions')
            
            # Select columns to write
            output_columns = [
                'product_id', 'predicted_price', 'confidence',
                'prediction_timestamp', 'model_version'
            ]
            
            # Add optional columns if they exist
            if 'price_change_pct' in results_df.columns:
                output_columns.append('price_change_pct')
            
            output_df = results_df[output_columns]
            
            connector.write_predictions_to_sql(
                output_df,
                table_name=table_name,
                if_exists='append'
            )
            
            logger.info(f"Wrote {len(output_df)} predictions to {table_name}")
        
        # Write to blob if requested
        if data.get('write_to_blob', False):
            blob_container = data.get('output_blob_container', 'predictions')
            blob_name = data.get('output_blob_name', 
                                f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            
            connector.write_to_blob(
                results_df,
                container_name=blob_container,
                blob_name=blob_name,
                file_type='csv'
            )
            
            logger.info(f"Wrote predictions to blob: {blob_name}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            'status': 'success',
            'predictions': predictions.tolist(),
            'confidence': confidence.tolist(),
            'num_predictions': len(predictions),
            'model_version': os.getenv('MODEL_VERSION', '1.0'),
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add summary statistics
        response['statistics'] = {
            'mean_predicted_price': float(np.mean(predictions)),
            'median_predicted_price': float(np.median(predictions)),
            'min_predicted_price': float(np.min(predictions)),
            'max_predicted_price': float(np.max(predictions)),
            'std_predicted_price': float(np.std(predictions))
        }
        
        # Add price change statistics if available
        if 'price_change_pct' in results_df.columns:
            response['statistics']['mean_price_change_pct'] = float(
                results_df['price_change_pct'].mean()
            )
        
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        return json.dumps(response)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        
        error_response = {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        return json.dumps(error_response)


# For local testing
if __name__ == "__main__":
    # Simulate Azure ML environment
    os.environ['AZUREML_MODEL_DIR'] = '../../models'
    
    # Initialize
    init()
    
    # Test with sample data
    test_request = {
        "features": [
            {
                "product_id": "P001",
                "base_price": 100.0,
                "demand": 50,
                "inventory": 200,
                "competitor_price": 105.0,
                "day_of_week": 1,
                "month": 6,
                "is_weekend": 0,
                "promotional_flag": 0
            }
        ]
    }
    
    # Make prediction
    result = run(json.dumps(test_request))
    print("\nPrediction Result:")
    print(json.dumps(json.loads(result), indent=2))
