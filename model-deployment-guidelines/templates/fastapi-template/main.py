"""
FastAPI Template for ML Model Deployment
Production-ready REST API template for any scikit-learn model
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production-ready ML model serving API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
feature_names = None


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input features for prediction"""
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature1": 100.0,
                    "feature2": 50,
                    "feature3": 200
                }
            }
        }


class BatchPredictionInput(BaseModel):
    """Batch prediction input"""
    instances: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class PredictionOutput(BaseModel):
    """Prediction response"""
    prediction: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: str


class BatchPredictionOutput(BaseModel):
    """Batch prediction response"""
    predictions: List[float]
    num_predictions: int
    timestamp: str
    model_version: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


@app.on_event("startup")
async def load_model():
    """Load model artifacts on startup"""
    global model, scaler, feature_names
    
    try:
        logger.info("Loading model artifacts...")
        
        model_dir = os.getenv('MODEL_DIR', './models')
        
        # Load model
        model_path = os.path.join(model_dir, 'model.pkl')
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler (optional)
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load feature names (optional)
        feature_names_path = os.path.join(model_dir, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def preprocess_features(features: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess input features
    
    Args:
        features: Dictionary of feature values
    
    Returns:
        Preprocessed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Select features if feature_names is defined
    if feature_names:
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        df = df[feature_names]
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Scale features if scaler is available
    if scaler:
        X = scaler.transform(df)
    else:
        X = df.values
    
    return X


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a single prediction
    
    Args:
        input_data: Input features
    
    Returns:
        Prediction result
    """
    try:
        # Preprocess features
        X = preprocess_features(input_data.features)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate confidence (if model supports it)
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            confidence = float(np.max(proba))
        
        return PredictionOutput(
            prediction=float(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def batch_predict(input_data: BatchPredictionInput):
    """
    Make batch predictions
    
    Args:
        input_data: List of feature dictionaries
    
    Returns:
        Batch prediction results
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(input_data.instances)
        
        # Select features if feature_names is defined
        if feature_names:
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            df = df[feature_names]
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Scale features if scaler is available
        if scaler:
            X = scaler.transform(df)
        else:
            X = df.values
        
        # Make predictions
        predictions = model.predict(X)
        
        return BatchPredictionOutput(
            predictions=predictions.tolist(),
            num_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information"""
    info = {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(feature_names) if feature_names else None,
        "feature_names": feature_names if feature_names else None,
        "model_type": type(model).__name__ if model else None
    }
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
