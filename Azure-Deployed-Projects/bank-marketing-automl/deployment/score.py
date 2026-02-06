"""
Scoring script for Bank Marketing model deployment.
This script is used by Azure ML to handle inference requests.
"""

import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model


def init():
    """
    This function is called when the container is initialized/started.
    Load the model once and keep it in memory for fast predictions.
    """
    global model
    
    try:
        # Get the path to the registered model
        model_path = Model.get_model_path('BankMarketingMLBest')
        
        # Load the model
        model = joblib.load(model_path)
        
        print("Model loaded successfully")
        print(f"Model type: {type(model)}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    
    Args:
        raw_data (str): JSON string containing input data
        
    Returns:
        str: JSON string with predictions and probabilities
        
    Example input:
    {
        "data": [{
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "duration": 180,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }]
    }
    
    Example output:
    {
        "predictions": [1],
        "probabilities": [[0.13, 0.87]],
        "prediction_labels": ["yes"]
    }
    """
    try:
        # Parse the input JSON
        data = json.loads(raw_data)
        input_data = data['data']
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(input_data)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Convert predictions to labels
        prediction_labels = ['yes' if p == 1 else 'no' for p in predictions]
        
        # Format the response
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'prediction_labels': prediction_labels,
            'num_samples': len(predictions)
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = str(e)
        print(f"Error during prediction: {error_message}")
        return json.dumps({
            'error': error_message,
            'status': 'failed'
        })
