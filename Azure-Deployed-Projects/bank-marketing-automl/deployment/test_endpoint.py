"""
Test script for Bank Marketing model endpoint.
This script demonstrates how to consume the deployed model API.
"""

import urllib.request
import json
import os


# Endpoint configuration
ENDPOINT_URL = 'https://mlportfolio-hwksp.uksouth.inference.ml.azure.com/score'
API_KEY = os.getenv('AZURE_ML_API_KEY', 'YOUR_API_KEY_HERE')


def predict_single_customer(customer_data):
    """
    Make a prediction for a single customer.
    
    Args:
        customer_data (dict): Customer features
        
    Returns:
        dict: Prediction result
    """
    # Prepare the data
    data = {
        "data": [customer_data]
    }
    
    # Convert to JSON
    body = str.encode(json.dumps(data))
    
    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    # Make the request
    req = urllib.request.Request(ENDPOINT_URL, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result
    except urllib.error.HTTPError as error:
        print(f"The request failed with status code: {error.code}")
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None


def predict_batch(customers_list):
    """
    Make predictions for multiple customers.
    
    Args:
        customers_list (list): List of customer feature dictionaries
        
    Returns:
        dict: Batch prediction results
    """
    data = {
        "data": customers_list
    }
    
    body = str.encode(json.dumps(data))
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    req = urllib.request.Request(ENDPOINT_URL, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result
    except urllib.error.HTTPError as error:
        print(f"The request failed with status code: {error.code}")
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None


# Example 1: High-probability customer (likely to subscribe)
high_prob_customer = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 5000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

# Example 2: Low-probability customer (unlikely to subscribe)
low_prob_customer = {
    "age": 22,
    "job": "student",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 100,
    "housing": "no",
    "loan": "yes",
    "contact": "telephone",
    "day": 10,
    "month": "jun",
    "duration": 30,
    "campaign": 10,
    "pdays": 5,
    "previous": 2,
    "poutcome": "failure"
}


if __name__ == "__main__":
    print("=" * 60)
    print("Bank Marketing Model - API Test")
    print("=" * 60)
    
    # Test 1: Single prediction (high probability)
    print("\nTest 1: High-probability customer")
    print("-" * 60)
    result = predict_single_customer(high_prob_customer)
    if result:
        print(f"Prediction: {result['prediction_labels'][0]}")
        print(f"Probability of 'yes': {result['probabilities'][0][1]:.2%}")
        print(f"Probability of 'no': {result['probabilities'][0][0]:.2%}")
    
    # Test 2: Single prediction (low probability)
    print("\nTest 2: Low-probability customer")
    print("-" * 60)
    result = predict_single_customer(low_prob_customer)
    if result:
        print(f"Prediction: {result['prediction_labels'][0]}")
        print(f"Probability of 'yes': {result['probabilities'][0][1]:.2%}")
        print(f"Probability of 'no': {result['probabilities'][0][0]:.2%}")
    
    # Test 3: Batch prediction
    print("\nTest 3: Batch prediction (2 customers)")
    print("-" * 60)
    result = predict_batch([high_prob_customer, low_prob_customer])
    if result:
        print(f"Number of predictions: {result['num_samples']}")
        for i, label in enumerate(result['prediction_labels']):
            prob_yes = result['probabilities'][i][1]
            print(f"Customer {i+1}: {label} (probability: {prob_yes:.2%})")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
