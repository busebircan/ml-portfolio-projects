# Deployment Guide - Bank Marketing Model

This guide explains how the model was deployed to Azure ML and how to replicate the deployment.

---

## Deployment Architecture

```
Azure ML Workspace
    ↓
Model Registry (BankMarketingMLBest:1)
    ↓
Managed Endpoint (mlportfolio-hwksp)
    ↓
REST API (https://mlportfolio-hwksp.uksouth.inference.ml.azure.com/score)
```

---

## Prerequisites

1. **Azure Subscription** with sufficient credits
2. **Azure ML Workspace** created
3. **Trained Model** registered in Model Registry
4. **Azure CLI** installed and configured

---

## Step-by-Step Deployment

### Step 1: Register the Model

After AutoML completes, register the best model:

```python
from azureml.core import Workspace, Model

# Connect to workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_name='BankMarketingMLBest',
    model_path='outputs/model.pkl',  # Path from training run
    description='XGBoost classifier for bank marketing prediction',
    tags={'algorithm': 'XGBoost', 'auc': '0.93'}
)

print(f"Model registered: {model.name} version {model.version}")
```

**Result:** Model appears in Azure ML Studio → Models

---

### Step 2: Create Scoring Script

Create `score.py` to handle inference requests:

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    """
    This function is called when the container is initialized/started.
    Load the model once and keep it in memory.
    """
    global model
    
    # Get model path
    model_path = Model.get_model_path('BankMarketingMLBest')
    
    # Load model
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    
    Args:
        raw_data: JSON string with input data
        
    Returns:
        JSON string with predictions
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        input_data = data['data']
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Format response
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error = str(e)
        return json.dumps({'error': error})
```

---

### Step 3: Create Environment Configuration

Create `environment.yml` for dependencies:

```yaml
name: bank-marketing-env
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - pip:
    - azureml-defaults
    - scikit-learn==1.0.2
    - xgboost==1.5.0
    - joblib==1.1.0
    - numpy==1.21.0
    - pandas==1.3.0
```

---

### Step 4: Create Inference Configuration

```python
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment

# Create environment
env = Environment.from_conda_specification(
    name='bank-marketing-env',
    file_path='environment.yml'
)

# Create inference config
inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)
```

---

### Step 5: Deploy to Managed Endpoint

```python
from azureml.core.webservice import AciWebservice

# Deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True,  # Enable key-based authentication
    enable_app_insights=True  # Enable monitoring
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name='mlportfolio-hwksp',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True  # Replace existing deployment
)

# Wait for deployment
service.wait_for_deployment(show_output=True)

print(f"Deployment state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
print(f"Swagger URI: {service.swagger_uri}")
```

**Deployment Time:** ~10 minutes

---

### Step 6: Get API Keys

```python
# Get primary and secondary keys
keys = service.get_keys()

print(f"Primary key: {keys['primaryKey']}")
print(f"Secondary key: {keys['secondaryKey']}")
```

**Security:** Store keys in Azure Key Vault, never commit to Git!

---

## Testing the Deployment

### Test 1: Simple Prediction

```python
import requests
import json

# Endpoint details
url = service.scoring_uri
api_key = keys['primaryKey']

# Sample customer data
data = {
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

# Make request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

response = requests.post(url, json=data, headers=headers)

print(f"Status: {response.status_code}")
print(f"Prediction: {response.json()}")
```

**Expected Output:**
```json
{
  "predictions": [1],
  "probabilities": [[0.13, 0.87]]
}
```

---

### Test 2: Batch Predictions

```python
# Multiple customers
batch_data = {
    "data": [
        {
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
        },
        {
            "age": 25,
            "job": "student",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 500,
            "housing": "no",
            "loan": "yes",
            "contact": "telephone",
            "day": 10,
            "month": "jun",
            "duration": 60,
            "campaign": 5,
            "pdays": 30,
            "previous": 1,
            "poutcome": "failure"
        }
    ]
}

response = requests.post(url, json=batch_data, headers=headers)
print(response.json())
```

---

## Monitoring

### View Metrics in Azure Portal

1. Navigate to Azure ML Studio
2. Go to Endpoints → mlportfolio-hwksp
3. Click "Monitoring" tab

**Available Metrics:**
- Request count
- Response time (latency)
- Error rate
- CPU/Memory usage

### Set Up Alerts

```python
from azureml.core.webservice import Webservice

# Get service
service = Webservice(workspace=ws, name='mlportfolio-hwksp')

# Enable Application Insights
service.update(enable_app_insights=True)

print("Application Insights enabled")
```

---

## Scaling

### Auto-Scaling Configuration

```python
from azureml.core.webservice import AksWebservice

# For production, use AKS instead of ACI
aks_config = AksWebservice.deploy_configuration(
    autoscale_enabled=True,
    autoscale_min_replicas=1,
    autoscale_max_replicas=10,
    autoscale_target_utilization=70,  # Scale when CPU > 70%
    cpu_cores=2,
    memory_gb=4
)
```

---

## Cost Optimization

### Current Costs (ACI Deployment)

- **Compute:** ~$0.10/hour (1 CPU, 1GB RAM)
- **Storage:** ~$0.05/month (model storage)
- **Requests:** Free (first 1M requests)

**Monthly Cost:** ~$75

### Cost Reduction Strategies

1. **Use Batch Endpoints** for non-real-time predictions
   - Cost: ~$0.01/hour (only when running)
   - Savings: 90%

2. **Use Spot Instances** for training
   - Cost: 60-80% cheaper than regular instances

3. **Delete Unused Endpoints**
   ```python
   service.delete()
   ```

---

## Troubleshooting

### Issue 1: Deployment Fails

**Error:** `Service deployment polling reached non-successful terminal state`

**Solution:**
1. Check logs: `service.get_logs()`
2. Verify environment.yml has correct dependencies
3. Test score.py locally before deploying

---

### Issue 2: High Latency

**Error:** Predictions take >1 second

**Solution:**
1. Load model in `init()`, not `run()`
2. Use faster instance type (2+ CPUs)
3. Enable caching for repeated requests

---

### Issue 3: Authentication Errors

**Error:** `401 Unauthorized`

**Solution:**
1. Verify API key is correct
2. Check key hasn't been regenerated
3. Ensure `Authorization: Bearer {key}` header is set

---

## Updating the Deployment

### Deploy New Model Version

```python
# Register new model
new_model = Model.register(
    workspace=ws,
    model_name='BankMarketingMLBest',
    model_path='outputs/model_v2.pkl'
)

# Update service
service.update(models=[new_model])
service.wait_for_deployment(show_output=True)

print(f"Updated to model version {new_model.version}")
```

---

## Rollback

### Revert to Previous Version

```python
# Get previous model version
old_model = Model(ws, name='BankMarketingMLBest', version=1)

# Rollback
service.update(models=[old_model])
service.wait_for_deployment(show_output=True)

print("Rolled back to version 1")
```

---

## Security Best Practices

1. **API Keys:**
   - Store in Azure Key Vault
   - Rotate keys regularly
   - Use secondary key for rotation without downtime

2. **Network Security:**
   - Enable private endpoints for production
   - Use VNet integration
   - Restrict IP addresses

3. **Data Security:**
   - Encrypt data in transit (HTTPS)
   - Encrypt data at rest
   - Log all requests for audit

---

## Next Steps

1. **Set up CI/CD** for automated retraining and deployment
2. **Implement A/B testing** to compare model versions
3. **Add data drift detection** to trigger retraining
4. **Create monitoring dashboard** for business metrics

---

## Resources

- [Azure ML Deployment Docs](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-and-where)
- [Managed Endpoints Guide](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-managed-online-endpoints)
- [Troubleshooting Guide](https://docs.microsoft.com/azure/machine-learning/how-to-troubleshoot-deployment)

---

**Deployment Status:** ✅ Production Ready

**Last Updated:** February 2026
