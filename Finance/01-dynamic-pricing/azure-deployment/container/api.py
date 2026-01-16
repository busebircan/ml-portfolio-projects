"""
FastAPI Application for Dynamic Pricing Model
Containerized REST API for Azure Container Instances
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_config import AzureDataConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dynamic Pricing API",
    description="ML-powered dynamic pricing optimization",
    version="1.0.0"
)

# Global variables
model = None
scaler = None
connector = None
feature_names = None


# Pydantic models for request/response
class PricingFeatures(BaseModel):
    """Individual product features for pricing"""
    product_id: str
    base_price: float = Field(..., gt=0, description="Base price of the product")
    demand: float = Field(..., ge=0, description="Current demand level")
    inventory: float = Field(..., ge=0, description="Available inventory")
    competitor_price: float = Field(..., gt=0, description="Competitor price")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    month: int = Field(..., ge=1, le=12, description="Month")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0 or 1)")
    promotional_flag: int = Field(..., ge=0, le=1, description="Promotion active (0 or 1)")


class DirectPredictionRequest(BaseModel):
    """Request with direct feature input"""
    features: List[PricingFeatures]
    write_to_db: bool = False
    output_table: Optional[str] = "price_predictions"


class SQLPredictionRequest(BaseModel):
    """Request with SQL query"""
    query: str
    params: Optional[List[Any]] = None
    write_to_db: bool = False
    output_table: Optional[str] = "price_predictions"


class BlobPredictionRequest(BaseModel):
    """Request with blob storage reference"""
    blob_container: str
    blob_name: str
    file_type: str = "csv"
    write_to_db: bool = False
    output_table: Optional[str] = "price_predictions"


class PredictionResponse(BaseModel):
    """Prediction response"""
    status: str
    predictions: List[float]
    confidence: List[float]
    num_predictions: int
    timestamp: str
    statistics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model, scaler, connector, feature_names
    
    logger.info("Loading model artifacts...")
    
    model_dir = os.getenv('MODEL_DIR', './models')
    
    try:
        model = joblib.load(os.path.join(model_dir, 'pricing_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load feature names
        with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Initialize Azure connector
        connector = AzureDataConnector(use_managed_identity=True)
        
        logger.info(f"Model loaded successfully with {len(feature_names)} features")
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Dynamic Pricing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_direct": "/predict/direct",
            "predict_sql": "/predict/sql",
            "predict_blob": "/predict/blob"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


def make_predictions(df: pd.DataFrame) -> tuple:
    """
    Make predictions on DataFrame
    
    Args:
        df: DataFrame with features
    
    Returns:
        Tuple of (predictions, confidence, results_df)
    """
    # Validate features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
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
    
    # Calculate price changes
    if 'base_price' in df.columns:
        results_df['price_change_pct'] = (
            (results_df['predicted_price'] - results_df['base_price']) / 
            results_df['base_price'] * 100
        )
    
    return predictions, confidence, results_df


@app.post("/predict/direct", response_model=PredictionResponse)
async def predict_direct(request: DirectPredictionRequest):
    """
    Make predictions with direct feature input
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([f.dict() for f in request.features])
        
        # Make predictions
        predictions, confidence, results_df = make_predictions(df)
        
        # Write to database if requested
        if request.write_to_db:
            output_columns = [
                'product_id', 'predicted_price', 'confidence', 'prediction_timestamp'
            ]
            
            if 'price_change_pct' in results_df.columns:
                output_columns.append('price_change_pct')
            
            connector.write_predictions_to_sql(
                results_df[output_columns],
                table_name=request.output_table,
                if_exists='append'
            )
        
        # Prepare response
        return PredictionResponse(
            status="success",
            predictions=predictions.tolist(),
            confidence=confidence.tolist(),
            num_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
            statistics={
                "mean_predicted_price": float(np.mean(predictions)),
                "median_predicted_price": float(np.median(predictions)),
                "min_predicted_price": float(np.min(predictions)),
                "max_predicted_price": float(np.max(predictions)),
                "std_predicted_price": float(np.std(predictions))
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/sql", response_model=PredictionResponse)
async def predict_sql(request: SQLPredictionRequest):
    """
    Make predictions from SQL query
    """
    try:
        # Load data from SQL
        df = connector.load_data_from_sql(request.query, params=request.params)
        
        # Make predictions
        predictions, confidence, results_df = make_predictions(df)
        
        # Write to database if requested
        if request.write_to_db:
            output_columns = [
                'product_id', 'predicted_price', 'confidence', 'prediction_timestamp'
            ]
            
            if 'price_change_pct' in results_df.columns:
                output_columns.append('price_change_pct')
            
            connector.write_predictions_to_sql(
                results_df[output_columns],
                table_name=request.output_table,
                if_exists='append'
            )
        
        # Prepare response
        return PredictionResponse(
            status="success",
            predictions=predictions.tolist(),
            confidence=confidence.tolist(),
            num_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
            statistics={
                "mean_predicted_price": float(np.mean(predictions)),
                "median_predicted_price": float(np.median(predictions)),
                "min_predicted_price": float(np.min(predictions)),
                "max_predicted_price": float(np.max(predictions))
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/blob", response_model=PredictionResponse)
async def predict_blob(request: BlobPredictionRequest):
    """
    Make predictions from blob storage
    """
    try:
        # Load data from blob
        df = connector.load_from_blob(
            container_name=request.blob_container,
            blob_name=request.blob_name,
            file_type=request.file_type
        )
        
        # Make predictions
        predictions, confidence, results_df = make_predictions(df)
        
        # Write to database if requested
        if request.write_to_db:
            output_columns = [
                'product_id', 'predicted_price', 'confidence', 'prediction_timestamp'
            ]
            
            if 'price_change_pct' in results_df.columns:
                output_columns.append('price_change_pct')
            
            connector.write_predictions_to_sql(
                results_df[output_columns],
                table_name=request.output_table,
                if_exists='append'
            )
        
        # Prepare response
        return PredictionResponse(
            status="success",
            predictions=predictions.tolist(),
            confidence=confidence.tolist(),
            num_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
            statistics={
                "mean_predicted_price": float(np.mean(predictions)),
                "median_predicted_price": float(np.median(predictions)),
                "min_predicted_price": float(np.min(predictions)),
                "max_predicted_price": float(np.max(predictions))
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
