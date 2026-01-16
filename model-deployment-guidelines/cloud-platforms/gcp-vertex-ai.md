# Google Cloud Vertex AI Deployment Guide

Complete guide for deploying ML models to Google Cloud Vertex AI with managed endpoints and batch predictions.

## Overview

Vertex AI is Google Cloud's unified ML platform that provides tools for building, deploying, and scaling ML models. This guide covers deploying scikit-learn models from this portfolio to Vertex AI endpoints.

---

## Prerequisites

### Required Tools
- Google Cloud SDK (gcloud CLI)
- Python 3.8+
- google-cloud-aiplatform SDK

### GCP Resources
- GCP Project with Vertex AI API enabled
- Service account with appropriate permissions
- Cloud Storage bucket for model artifacts

### Installation

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Install Python dependencies
pip install google-cloud-aiplatform google-cloud-storage scikit-learn
```

---

## Step 1: Enable APIs and Set Up Project

```bash
# Set project
export PROJECT_ID="your-project-id"
export REGION="us-central1"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com

# Create service account
gcloud iam service-accounts create vertex-ai-sa \
    --display-name="Vertex AI Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vertex-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

---

## Step 2: Prepare Custom Prediction Container

### Create Predictor Class

```python
# predictor.py
import os
import joblib
import numpy as np
from google.cloud import storage

class PricingPredictor:
    """Custom predictor for Vertex AI"""
    
    def __init__(self):
        """Load model artifacts"""
        self.model = None
        self.scaler = None
    
    def load(self, artifacts_uri: str):
        """Load model from Cloud Storage"""
        # Download model files
        storage_client = storage.Client()
        
        # Parse GCS URI
        bucket_name = artifacts_uri.split('/')[2]
        prefix = '/'.join(artifacts_uri.split('/')[3:])
        
        bucket = storage_client.bucket(bucket_name)
        
        # Download model
        blob = bucket.blob(f'{prefix}/model.pkl')
        blob.download_to_filename('/tmp/model.pkl')
        self.model = joblib.load('/tmp/model.pkl')
        
        # Download scaler
        blob = bucket.blob(f'{prefix}/scaler.pkl')
        blob.download_to_filename('/tmp/scaler.pkl')
        self.scaler = joblib.load('/tmp/scaler.pkl')
        
        print("Model loaded successfully")
    
    def preprocess(self, instances):
        """Preprocess input data"""
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(instances)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def predict(self, instances):
        """Make predictions"""
        # Preprocess
        X = self.preprocess(instances)
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions.tolist()
    
    def postprocess(self, predictions):
        """Format predictions"""
        return {
            'predictions': predictions
        }
```

### Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy predictor code
COPY predictor.py .

# Set environment variables
ENV AIP_STORAGE_URI=""
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Run predictor server
CMD exec python -m google.cloud.aiplatform.prediction.predictor \
    --predictor_class=predictor.PricingPredictor \
    --http_port=$AIP_HTTP_PORT \
    --health_route=$AIP_HEALTH_ROUTE \
    --predict_route=$AIP_PREDICT_ROUTE
```

### Requirements

```txt
# requirements.txt
google-cloud-aiplatform>=1.25.0
google-cloud-storage>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.1.0
```

---

## Step 3: Build and Push Container

```bash
# Set variables
export IMAGE_NAME="pricing-predictor"
export IMAGE_TAG="latest"
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build container
docker build -t $IMAGE_URI .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to Container Registry
docker push $IMAGE_URI

echo "Container pushed: $IMAGE_URI"
```

---

## Step 4: Upload Model to Cloud Storage

```python
# upload_model.py
from google.cloud import storage
import os

def upload_model_artifacts(bucket_name, model_dir, destination_prefix):
    """Upload model artifacts to GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)
        blob = bucket.blob(f'{destination_prefix}/{file_name}')
        blob.upload_from_filename(file_path)
        print(f'Uploaded {file_name}')
    
    return f'gs://{bucket_name}/{destination_prefix}'

if __name__ == "__main__":
    model_uri = upload_model_artifacts(
        bucket_name='your-bucket-name',
        model_dir='./models',
        destination_prefix='models/pricing-model'
    )
    print(f'Model URI: {model_uri}')
```

---

## Step 5: Deploy to Vertex AI

```python
# deploy_to_vertex.py
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='your-project-id',
    location='us-central1'
)

# Upload model
model = aiplatform.Model.upload(
    display_name='pricing-model',
    artifact_uri='gs://your-bucket/models/pricing-model',
    serving_container_image_uri='gcr.io/your-project-id/pricing-predictor:latest',
    serving_container_predict_route='/predict',
    serving_container_health_route='/health',
    serving_container_ports=[8080]
)

print(f'Model uploaded: {model.resource_name}')

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name='pricing-endpoint'
)

print(f'Endpoint created: {endpoint.resource_name}')

# Deploy model to endpoint
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='pricing-v1',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=5,
    traffic_percentage=100
)

print(f'Model deployed to endpoint: {endpoint.resource_name}')
```

---

## Step 6: Make Predictions

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(
    project='your-project-id',
    location='us-central1'
)

# Get endpoint
endpoint = aiplatform.Endpoint('projects/PROJECT_NUM/locations/REGION/endpoints/ENDPOINT_ID')

# Prepare test data
test_instances = [
    {
        'product_id': 'P001',
        'base_price': 100.0,
        'demand': 50,
        'inventory': 200,
        'competitor_price': 105.0,
        'day_of_week': 1,
        'month': 6,
        'is_weekend': 0,
        'promotional_flag': 0
    }
]

# Make prediction
predictions = endpoint.predict(instances=test_instances)

print(f'Predictions: {predictions.predictions}')
```

---

## Batch Predictions

```python
from google.cloud import aiplatform

# Create batch prediction job
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name='pricing-batch-prediction',
    model_name='projects/PROJECT_NUM/locations/REGION/models/MODEL_ID',
    instances_format='jsonl',
    gcs_source='gs://your-bucket/input/batch_input.jsonl',
    gcs_destination_prefix='gs://your-bucket/output/',
    machine_type='n1-standard-4',
    starting_replica_count=1,
    max_replica_count=5
)

# Wait for completion
batch_prediction_job.wait()

print(f'Batch prediction completed: {batch_prediction_job.resource_name}')
```

---

## Complete Deployment Script

```python
# complete_deployment.py
from google.cloud import aiplatform, storage
import os

class VertexAIDeployer:
    """Complete Vertex AI deployment workflow"""
    
    def __init__(self, project_id, location, bucket_name):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        
        aiplatform.init(project=project_id, location=location)
        self.storage_client = storage.Client()
    
    def upload_model_artifacts(self, model_dir, model_name):
        """Upload model to GCS"""
        bucket = self.storage_client.bucket(self.bucket_name)
        prefix = f'models/{model_name}'
        
        for file_name in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file_name)
            blob = bucket.blob(f'{prefix}/{file_name}')
            blob.upload_from_filename(file_path)
            print(f'✓ Uploaded {file_name}')
        
        return f'gs://{self.bucket_name}/{prefix}'
    
    def upload_model(self, display_name, artifact_uri, container_uri):
        """Upload model to Vertex AI"""
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=container_uri,
            serving_container_predict_route='/predict',
            serving_container_health_route='/health',
            serving_container_ports=[8080]
        )
        print(f'✓ Model uploaded: {model.display_name}')
        return model
    
    def create_endpoint(self, display_name):
        """Create Vertex AI endpoint"""
        endpoint = aiplatform.Endpoint.create(display_name=display_name)
        print(f'✓ Endpoint created: {endpoint.display_name}')
        return endpoint
    
    def deploy_model(self, model, endpoint, machine_type='n1-standard-4'):
        """Deploy model to endpoint"""
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f'{model.display_name}-v1',
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=5,
            traffic_percentage=100
        )
        print(f'✓ Model deployed to endpoint')
        return endpoint
    
    def full_deployment(self, model_dir, model_name, container_uri):
        """Complete deployment workflow"""
        print("=" * 60)
        print("VERTEX AI DEPLOYMENT")
        print("=" * 60)
        
        print("\n1. Uploading model artifacts...")
        artifact_uri = self.upload_model_artifacts(model_dir, model_name)
        
        print("\n2. Uploading model to Vertex AI...")
        model = self.upload_model(model_name, artifact_uri, container_uri)
        
        print("\n3. Creating endpoint...")
        endpoint = self.create_endpoint(f'{model_name}-endpoint')
        
        print("\n4. Deploying model...")
        endpoint = self.deploy_model(model, endpoint)
        
        print("\n" + "=" * 60)
        print("DEPLOYMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"\nEndpoint ID: {endpoint.name}")
        print(f"Endpoint URL: {endpoint.gca_resource.display_name}")
        
        return endpoint

if __name__ == "__main__":
    deployer = VertexAIDeployer(
        project_id='your-project-id',
        location='us-central1',
        bucket_name='your-bucket-name'
    )
    
    endpoint = deployer.full_deployment(
        model_dir='./models',
        model_name='pricing-model',
        container_uri='gcr.io/your-project-id/pricing-predictor:latest'
    )
```

---

## Monitoring

```python
from google.cloud import monitoring_v3

# Create monitoring client
client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Query prediction latency
interval = monitoring_v3.TimeInterval(
    {
        "end_time": {"seconds": int(time.time())},
        "start_time": {"seconds": int(time.time()) - 3600},
    }
)

results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="aiplatform.googleapis.com/prediction/online/prediction_latencies"',
        "interval": interval,
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    }
)

for result in results:
    print(result)
```

---

## Cost Optimization

### Pricing
- **n1-standard-4:** ~$0.19/hour
- **n1-standard-2:** ~$0.095/hour
- **Batch predictions:** ~$0.08/hour per node

### Tips
- Use auto-scaling with appropriate min/max replicas
- Use batch predictions for scheduled workloads
- Delete unused endpoints
- Use preemptible VMs for batch jobs

---

## Cleanup

```python
# Undeploy model
endpoint.undeploy_all()

# Delete endpoint
endpoint.delete()

# Delete model
model.delete()

print("Resources cleaned up")
```

---

**Last Updated:** January 2026
