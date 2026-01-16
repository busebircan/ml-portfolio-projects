# AWS SageMaker Deployment Guide

Complete guide for deploying ML models to Amazon SageMaker with real-time and batch inference capabilities.

## Overview

Amazon SageMaker is a fully managed service that provides tools to build, train, and deploy machine learning models at scale. This guide covers deploying models from this portfolio to SageMaker endpoints.

---

## Prerequisites

### Required Tools
- AWS CLI (version 2.0+)
- Python 3.8+
- boto3 (AWS SDK for Python)
- sagemaker Python SDK

### AWS Resources
- AWS Account with appropriate IAM permissions
- S3 bucket for model artifacts
- IAM role for SageMaker execution

### Installation

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Python dependencies
pip install boto3 sagemaker scikit-learn pandas joblib
```

---

## Step 1: Prepare Model for SageMaker

### Model Artifact Structure

SageMaker requires models to be packaged in a specific format.

```
model.tar.gz
├── model.pkl          # Trained model
├── scaler.pkl         # Preprocessing artifacts
└── inference.py       # Inference script
```

### Create Inference Script

```python
# inference.py
import json
import joblib
import pandas as pd
import numpy as np
import os

def model_fn(model_dir):
    """Load model from the model directory"""
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    return {'model': model, 'scaler': scaler}

def input_fn(request_body, content_type='application/json'):
    """Parse input data"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame(data['features'])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    # Scale features
    X_scaled = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

def output_fn(predictions, accept='application/json'):
    """Format output"""
    if accept == 'application/json':
        return json.dumps({
            'predictions': predictions.tolist()
        }), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

### Package Model

```python
# package_model.py
import tarfile
import os

def package_model(model_dir, output_path='model.tar.gz'):
    """Package model artifacts for SageMaker"""
    with tarfile.open(output_path, 'w:gz') as tar:
        tar.add(os.path.join(model_dir, 'model.pkl'), arcname='model.pkl')
        tar.add(os.path.join(model_dir, 'scaler.pkl'), arcname='scaler.pkl')
        tar.add('inference.py', arcname='inference.py')
    
    print(f"Model packaged: {output_path}")

if __name__ == "__main__":
    package_model('./models')
```

---

## Step 2: Upload Model to S3

```python
import boto3

s3_client = boto3.client('s3')
bucket_name = 'your-sagemaker-bucket'
model_key = 'models/pricing-model/model.tar.gz'

# Upload model
s3_client.upload_file(
    'model.tar.gz',
    bucket_name,
    model_key
)

print(f"Model uploaded to s3://{bucket_name}/{model_key}")
```

---

## Step 3: Create SageMaker Model

```python
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole'

# Model location in S3
model_data = f's3://{bucket_name}/{model_key}'

# Create SKLearn model
sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',
    framework_version='1.0-1',
    py_version='py3',
    name='pricing-model'
)

print("SageMaker model created")
```

---

## Step 4: Deploy Real-Time Endpoint

```python
# Deploy to real-time endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='pricing-endpoint'
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

---

## Step 5: Make Predictions

```python
import json

# Prepare test data
test_data = {
    'features': [
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
}

# Make prediction
response = predictor.predict(test_data)
print(f"Prediction: {response}")
```

---

## Batch Transform

For batch predictions on large datasets.

```python
from sagemaker.sklearn import SKLearnModel

# Create transformer
transformer = sklearn_model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket_name}/batch-predictions/'
)

# Run batch transform
transformer.transform(
    data=f's3://{bucket_name}/input-data/batch_input.csv',
    content_type='text/csv',
    split_type='Line'
)

transformer.wait()
print("Batch transform completed")
```

---

## Auto-Scaling

Configure auto-scaling for variable workloads.

```python
import boto3

client = boto3.client('application-autoscaling')

# Register scalable target
response = client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Create scaling policy
response = client.put_scaling_policy(
    PolicyName='pricing-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)

print("Auto-scaling configured")
```

---

## Monitoring with CloudWatch

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create alarm for high latency
cloudwatch.put_metric_alarm(
    AlarmName='pricing-high-latency',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Period=300,
    Statistic='Average',
    Threshold=1000.0,  # 1 second
    ActionsEnabled=True,
    AlarmDescription='Alert when model latency exceeds 1 second',
    Dimensions=[
        {
            'Name': 'EndpointName',
            'Value': predictor.endpoint_name
        },
        {
            'Name': 'VariantName',
            'Value': 'AllTraffic'
        }
    ]
)
```

---

## Cost Optimization

### Pricing
- **ml.m5.large:** ~$0.115/hour
- **ml.t3.medium:** ~$0.05/hour (for low-volume)
- **Batch Transform:** Pay only for processing time

### Tips
- Use smaller instances for low-volume endpoints
- Implement auto-scaling to match demand
- Use batch transform for scheduled predictions
- Delete unused endpoints

---

## Complete Deployment Script

```python
# deploy_to_sagemaker.py
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
import tarfile
import os

class SageMakerDeployer:
    def __init__(self, role, bucket_name):
        self.role = role
        self.bucket_name = bucket_name
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client('s3')
    
    def package_model(self, model_dir, output_path='model.tar.gz'):
        """Package model artifacts"""
        with tarfile.open(output_path, 'w:gz') as tar:
            for file in os.listdir(model_dir):
                tar.add(os.path.join(model_dir, file), arcname=file)
            tar.add('inference.py', arcname='inference.py')
        return output_path
    
    def upload_model(self, local_path, s3_key):
        """Upload model to S3"""
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        return f's3://{self.bucket_name}/{s3_key}'
    
    def create_model(self, model_data, model_name):
        """Create SageMaker model"""
        model = SKLearnModel(
            model_data=model_data,
            role=self.role,
            entry_point='inference.py',
            framework_version='1.0-1',
            py_version='py3',
            name=model_name
        )
        return model
    
    def deploy_endpoint(self, model, endpoint_name, instance_type='ml.m5.large'):
        """Deploy real-time endpoint"""
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        return predictor
    
    def full_deployment(self, model_dir, model_name, endpoint_name):
        """Complete deployment workflow"""
        print("Step 1: Packaging model...")
        package_path = self.package_model(model_dir)
        
        print("Step 2: Uploading to S3...")
        s3_key = f'models/{model_name}/model.tar.gz'
        model_data = self.upload_model(package_path, s3_key)
        
        print("Step 3: Creating SageMaker model...")
        model = self.create_model(model_data, model_name)
        
        print("Step 4: Deploying endpoint...")
        predictor = self.deploy_endpoint(model, endpoint_name)
        
        print(f"✓ Deployment complete!")
        print(f"  Endpoint: {predictor.endpoint_name}")
        return predictor

if __name__ == "__main__":
    deployer = SageMakerDeployer(
        role='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole',
        bucket_name='your-sagemaker-bucket'
    )
    
    predictor = deployer.full_deployment(
        model_dir='./models',
        model_name='pricing-model',
        endpoint_name='pricing-endpoint'
    )
```

---

## Cleanup

```python
# Delete endpoint
predictor.delete_endpoint()

# Delete model
predictor.delete_model()

print("Resources cleaned up")
```

---

## Next Steps

- Set up CI/CD with AWS CodePipeline
- Implement A/B testing with multiple variants
- Add model monitoring with SageMaker Model Monitor
- Integrate with AWS Lambda for event-driven predictions

---

**Last Updated:** January 2026
