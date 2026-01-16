#!/bin/bash

# Azure Container Deployment Script
# Deploys Dynamic Pricing model as a containerized REST API

set -e

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-pricing-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-pricingacr}"
CONTAINER_NAME="${CONTAINER_NAME:-pricing-api}"
IMAGE_NAME="dynamic-pricing-api"
IMAGE_TAG="latest"

echo "======================================"
echo "Azure Container Deployment"
echo "======================================"
echo ""
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "ACR Name: $ACR_NAME"
echo "Container Name: $CONTAINER_NAME"
echo ""

# Step 1: Create resource group
echo "Step 1: Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# Step 2: Create Azure Container Registry
echo ""
echo "Step 2: Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Step 3: Build and push Docker image
echo ""
echo "Step 3: Building and pushing Docker image..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME:$IMAGE_TAG \
    --file Dockerfile \
    .

# Step 4: Get ACR credentials
echo ""
echo "Step 4: Retrieving ACR credentials..."
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

# Step 5: Deploy to Azure Container Instances
echo ""
echo "Step 5: Deploying to Azure Container Instances..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label $CONTAINER_NAME \
    --ports 8000 \
    --cpu 2 \
    --memory 4 \
    --environment-variables \
        MODEL_DIR=/app/models \
    --restart-policy Always

# Step 6: Get container details
echo ""
echo "Step 6: Retrieving container details..."
FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --query ipAddress.fqdn -o tsv)

echo ""
echo "======================================"
echo "Deployment Successful!"
echo "======================================"
echo ""
echo "API URL: http://$FQDN:8000"
echo "Health Check: http://$FQDN:8000/health"
echo "API Docs: http://$FQDN:8000/docs"
echo ""
echo "Test the API:"
echo "curl http://$FQDN:8000/health"
echo ""
