# Azure Deployment Guide - Dynamic Pricing Model

Complete guide for deploying the Dynamic Pricing machine learning model to Microsoft Azure with database integration.

## Table of Contents

1. [Overview](#overview)
2. [Deployment Options](#deployment-options)
3. [Prerequisites](#prerequisites)
4. [Azure Configuration](#azure-configuration)
5. [Option 1: Azure ML Studio](#option-1-azure-ml-studio)
6. [Option 2: Azure Functions](#option-2-azure-functions)
7. [Option 3: Azure Container Instances](#option-3-azure-container-instances)
8. [Database Integration](#database-integration)
9. [Testing the Deployment](#testing-the-deployment)
10. [Monitoring and Maintenance](#monitoring-and-maintenance)
11. [Cost Optimization](#cost-optimization)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This deployment package provides three production-ready options for deploying the Dynamic Pricing model to Azure, each with database integration capabilities for Azure SQL Database, Cosmos DB, and Blob Storage.

### Architecture Diagram

```
┌─────────────────┐
│  Azure SQL DB   │◄──────┐
└─────────────────┘       │
                          │
┌─────────────────┐       │    ┌──────────────────┐
│  Blob Storage   │◄──────┼────│  Pricing Model   │
└─────────────────┘       │    │   (Deployed)     │
                          │    └──────────────────┘
┌─────────────────┐       │              │
│  Key Vault      │◄──────┘              │
└─────────────────┘                      │
                                         ▼
                              ┌──────────────────┐
                              │  Client Apps     │
                              └──────────────────┘
```

---

## Deployment Options

### Comparison Matrix

| Feature | Azure ML Studio | Azure Functions | Container Instances |
|---------|----------------|-----------------|---------------------|
| **Use Case** | Production ML workloads | Serverless, event-driven | Full control, custom apps |
| **Scaling** | Auto-scale | Auto-scale | Manual scale |
| **Cold Start** | No | Yes (~5s) | No |
| **Cost** | $$$ | $ | $$ |
| **Complexity** | Medium | Low | High |
| **Best For** | High-volume predictions | Sporadic requests | Custom requirements |

---

## Prerequisites

### Required Tools

- **Azure CLI** (version 2.40+)
- **Python** (3.9+)
- **Docker** (for container deployment)
- **Git**

### Azure Resources

- **Azure Subscription** with contributor access
- **Resource Group** for deployment
- **Azure SQL Database** (optional, for data storage)
- **Azure Key Vault** (recommended for secrets)

### Installation

Install Azure CLI on your local machine.

**Windows:**
```bash
winget install Microsoft.AzureCLI
```

**macOS:**
```bash
brew install azure-cli
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Authentication

Log in to Azure and set your subscription.

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Verify
az account show
```

---

## Azure Configuration

### Step 1: Create Resource Group

Create a dedicated resource group for the deployment.

```bash
az group create \
    --name pricing-ml-rg \
    --location eastus
```

### Step 2: Create Azure SQL Database

Set up Azure SQL Database for storing features and predictions.

```bash
# Create SQL Server
az sql server create \
    --name pricing-sql-server \
    --resource-group pricing-ml-rg \
    --location eastus \
    --admin-user sqladmin \
    --admin-password "YourSecurePassword123!"

# Create Database
az sql db create \
    --resource-group pricing-ml-rg \
    --server pricing-sql-server \
    --name pricing-db \
    --service-objective S0

# Configure firewall
az sql server firewall-rule create \
    --resource-group pricing-ml-rg \
    --server pricing-sql-server \
    --name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0
```

### Step 3: Create Key Vault

Store secrets securely in Azure Key Vault.

```bash
# Create Key Vault
az keyvault create \
    --name pricing-keyvault \
    --resource-group pricing-ml-rg \
    --location eastus

# Store SQL credentials
az keyvault secret set \
    --vault-name pricing-keyvault \
    --name SQL-USERNAME \
    --value "sqladmin"

az keyvault secret set \
    --vault-name pricing-keyvault \
    --name SQL-PASSWORD \
    --value "YourSecurePassword123!"
```

### Step 4: Create Storage Account

Set up Blob Storage for data files and model artifacts.

```bash
# Create storage account
az storage account create \
    --name pricingstorage \
    --resource-group pricing-ml-rg \
    --location eastus \
    --sku Standard_LRS

# Create containers
az storage container create \
    --name training-data \
    --account-name pricingstorage

az storage container create \
    --name predictions \
    --account-name pricingstorage
```

### Step 5: Set Environment Variables

Configure local environment variables for deployment scripts.

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="pricing-ml-rg"
export AZURE_WORKSPACE_NAME="pricing-ml-workspace"
export AZURE_SQL_SERVER="pricing-sql-server.database.windows.net"
export AZURE_SQL_DATABASE="pricing-db"
export KEY_VAULT_URL="https://pricing-keyvault.vault.azure.net/"
```

---

## Option 1: Azure ML Studio

Deploy the model as a managed online endpoint with auto-scaling and monitoring.

### Step 1: Create Azure ML Workspace

```bash
az ml workspace create \
    --name pricing-ml-workspace \
    --resource-group pricing-ml-rg \
    --location eastus
```

### Step 2: Install Azure ML SDK

```bash
cd ml-studio
pip install -r requirements-azure.txt
```

### Step 3: Prepare Model Files

Ensure your trained model is in the correct directory structure.

```
models/
├── pricing_model.pkl
├── scaler.pkl
└── feature_names.txt
```

### Step 4: Run Deployment Script

Execute the deployment script to register the model and create the endpoint.

```bash
python deploy_to_azure_ml.py
```

The script performs the following actions:
- Registers the model in Azure ML
- Creates a managed environment
- Creates an online endpoint
- Deploys the model
- Sets traffic to 100%
- Tests the endpoint

### Step 5: Get Endpoint Details

Retrieve the scoring URI and authentication key.

```bash
# Get scoring URI
az ml online-endpoint show \
    --name pricing-endpoint \
    --resource-group pricing-ml-rg \
    --workspace-name pricing-ml-workspace \
    --query scoring_uri

# Get keys
az ml online-endpoint get-credentials \
    --name pricing-endpoint \
    --resource-group pricing-ml-rg \
    --workspace-name pricing-ml-workspace
```

### Step 6: Test the Endpoint

Test the deployed endpoint with sample data.

```python
import requests
import json

endpoint_uri = "https://pricing-endpoint.eastus.inference.ml.azure.com/score"
api_key = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
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

response = requests.post(endpoint_uri, headers=headers, json=data)
print(response.json())
```

---

## Option 2: Azure Functions

Deploy as a serverless function for cost-effective, event-driven predictions.

### Step 1: Install Azure Functions Core Tools

```bash
# Windows
npm install -g azure-functions-core-tools@4

# macOS
brew tap azure/functions
brew install azure-functions-core-tools@4

# Linux
wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install azure-functions-core-tools-4
```

### Step 2: Create Function App

```bash
az functionapp create \
    --resource-group pricing-ml-rg \
    --consumption-plan-location eastus \
    --runtime python \
    --runtime-version 3.9 \
    --functions-version 4 \
    --name pricing-function-app \
    --storage-account pricingstorage \
    --os-type Linux
```

### Step 3: Configure Application Settings

Set environment variables for the function app.

```bash
az functionapp config appsettings set \
    --name pricing-function-app \
    --resource-group pricing-ml-rg \
    --settings \
        AZURE_SQL_SERVER="pricing-sql-server.database.windows.net" \
        AZURE_SQL_DATABASE="pricing-db" \
        KEY_VAULT_URL="https://pricing-keyvault.vault.azure.net/" \
        MODEL_DIR="/home/site/wwwroot/models"
```

### Step 4: Deploy Function

Navigate to the functions directory and deploy.

```bash
cd functions
func azure functionapp publish pricing-function-app
```

### Step 5: Test Function

Get the function URL and test it.

```bash
# Get function URL
FUNCTION_URL=$(az functionapp function show \
    --name pricing-function-app \
    --resource-group pricing-ml-rg \
    --function-name predict \
    --query invokeUrlTemplate -o tsv)

# Test
curl -X POST $FUNCTION_URL \
    -H "Content-Type: application/json" \
    -d '{
        "features": [{
            "product_id": "P001",
            "base_price": 100.0,
            "demand": 50,
            "inventory": 200,
            "competitor_price": 105.0,
            "day_of_week": 1,
            "month": 6,
            "is_weekend": 0,
            "promotional_flag": 0
        }]
    }'
```

---

## Option 3: Azure Container Instances

Deploy as a containerized REST API with full control over the environment.

### Step 1: Build Docker Image Locally (Optional)

Test the Docker image locally before deploying.

```bash
cd container

# Build image
docker build -t pricing-api:latest .

# Run locally
docker run -p 8000:8000 \
    -e AZURE_SQL_SERVER="your-server" \
    -e AZURE_SQL_DATABASE="your-db" \
    pricing-api:latest

# Test
curl http://localhost:8000/health
```

### Step 2: Deploy to Azure

Use the provided deployment script.

```bash
# Set environment variables
export AZURE_RESOURCE_GROUP="pricing-ml-rg"
export AZURE_LOCATION="eastus"
export AZURE_ACR_NAME="pricingacr"
export CONTAINER_NAME="pricing-api"

# Run deployment
./deploy_container.sh
```

The script performs the following:
- Creates Azure Container Registry
- Builds and pushes Docker image
- Deploys to Azure Container Instances
- Exposes API on port 8000

### Step 3: Access API

The deployment script outputs the API URL. Access the interactive API documentation.

```
API URL: http://pricing-api.eastus.azurecontainer.io:8000
API Docs: http://pricing-api.eastus.azurecontainer.io:8000/docs
```

---

## Database Integration

### Azure SQL Database

#### Create Tables

Connect to Azure SQL Database and create the necessary tables.

```sql
-- Features table
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    base_price DECIMAL(10,2),
    demand DECIMAL(10,2),
    inventory INT,
    competitor_price DECIMAL(10,2),
    date DATE,
    day_of_week INT,
    month INT,
    is_weekend INT,
    promotional_flag INT
);

-- Predictions table
CREATE TABLE price_predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    product_id VARCHAR(50),
    predicted_price DECIMAL(10,2),
    confidence DECIMAL(5,4),
    price_change_pct DECIMAL(10,2),
    prediction_timestamp DATETIME,
    model_version VARCHAR(20),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Index for performance
CREATE INDEX idx_predictions_timestamp 
ON price_predictions(prediction_timestamp DESC);
```

#### Query Data for Predictions

Use SQL queries to fetch data for predictions.

```python
from azure_config import AzureDataConnector

connector = AzureDataConnector()

# Load products needing predictions
query = """
    SELECT product_id, base_price, demand, inventory, 
           competitor_price, day_of_week, month, 
           is_weekend, promotional_flag
    FROM products
    WHERE date = CAST(GETDATE() AS DATE)
"""

df = connector.load_data_from_sql(query)
```

### Blob Storage

#### Upload Training Data

Store historical data in Blob Storage for model retraining.

```python
from azure_config import AzureDataConnector
import pandas as pd

connector = AzureDataConnector()

# Load historical data
df = pd.read_csv('historical_prices.csv')

# Upload to blob
connector.write_to_blob(
    df=df,
    container_name='training-data',
    blob_name='historical_prices.csv',
    file_type='csv'
)
```

---

## Testing the Deployment

### Test with Direct Features

Send a POST request with direct feature input.

```python
import requests
import json

url = "your-endpoint-url"
headers = {"Content-Type": "application/json"}

data = {
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

response = requests.post(url, headers=headers, json=data)
print(json.dumps(response.json(), indent=2))
```

### Test with SQL Query

Query data from Azure SQL and make predictions.

```python
data = {
    "query": "SELECT * FROM products WHERE date = '2024-01-15'",
    "write_to_db": True,
    "output_table": "price_predictions"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### Load Testing

Use Apache Bench or similar tools to test performance.

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p test_data.json -T application/json \
   http://your-endpoint-url/predict
```

---

## Monitoring and Maintenance

### Enable Application Insights

Monitor endpoint performance and errors.

```bash
# Create Application Insights
az monitor app-insights component create \
    --app pricing-insights \
    --location eastus \
    --resource-group pricing-ml-rg

# Link to Function App or Container
az functionapp config appsettings set \
    --name pricing-function-app \
    --resource-group pricing-ml-rg \
    --settings APPINSIGHTS_INSTRUMENTATIONKEY="your-key"
```

### Set Up Alerts

Create alerts for critical metrics.

```bash
# Alert on high error rate
az monitor metrics alert create \
    --name high-error-rate \
    --resource-group pricing-ml-rg \
    --scopes /subscriptions/{sub-id}/resourceGroups/pricing-ml-rg \
    --condition "avg Percentage CPU > 80" \
    --description "Alert when CPU exceeds 80%"
```

### Model Monitoring

Track model performance metrics over time.

```sql
-- Query prediction accuracy
SELECT 
    DATE(prediction_timestamp) as date,
    AVG(confidence) as avg_confidence,
    COUNT(*) as num_predictions
FROM price_predictions
GROUP BY DATE(prediction_timestamp)
ORDER BY date DESC;
```

---

## Cost Optimization

### Pricing Comparison

**Azure ML Studio:**
- Compute: ~$0.50/hour (Standard_DS3_v2)
- Storage: ~$0.05/GB/month
- Estimated monthly: $360 (24/7 operation)

**Azure Functions:**
- Execution: $0.20 per million executions
- Compute: $0.000016/GB-second
- Estimated monthly: $5-50 (depending on usage)

**Container Instances:**
- Compute: ~$0.10/hour (2 vCPU, 4GB RAM)
- Estimated monthly: $72 (24/7 operation)

### Cost Reduction Tips

**Use Azure Functions for sporadic workloads.** Functions scale to zero when not in use, eliminating idle costs.

**Schedule batch predictions.** Run predictions during off-peak hours using Azure Logic Apps or scheduled functions.

**Use spot instances.** For non-critical workloads, use spot instances to save up to 90% on compute costs.

**Implement caching.** Cache frequent predictions to reduce compute requirements.

---

## Troubleshooting

### Common Issues

**Issue: Model fails to load**

Check that model files are in the correct location and format.

```bash
# Verify model files
ls -lh models/
# Should show: pricing_model.pkl, scaler.pkl, feature_names.txt
```

**Issue: Database connection fails**

Verify firewall rules and connection strings.

```bash
# Test SQL connection
sqlcmd -S pricing-sql-server.database.windows.net \
       -d pricing-db \
       -U sqladmin \
       -P "YourPassword"
```

**Issue: Endpoint returns 500 error**

Check application logs for detailed error messages.

```bash
# Azure ML logs
az ml online-deployment get-logs \
    --name blue \
    --endpoint-name pricing-endpoint

# Function logs
az functionapp log tail \
    --name pricing-function-app \
    --resource-group pricing-ml-rg
```

### Debug Mode

Enable verbose logging for troubleshooting.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Next Steps

**Implement CI/CD pipeline** using Azure DevOps or GitHub Actions for automated deployments.

**Add model versioning** to track and roll back model changes.

**Set up A/B testing** to compare model versions in production.

**Implement data drift detection** to monitor input data quality.

**Add authentication** using Azure AD for enhanced security.

---

## Support

For issues or questions regarding this deployment package, please refer to the main project repository or contact the development team.

**Repository:** https://github.com/busebircan/ml-portfolio-projects

---

**Last Updated:** January 2026  
**Version:** 1.0.0
