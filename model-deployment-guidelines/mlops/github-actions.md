# GitHub Actions CI/CD for ML Models

Complete guide for setting up automated deployment pipelines using GitHub Actions.

## Overview

GitHub Actions provides integrated CI/CD capabilities for automating model training, testing, and deployment workflows directly from your repository.

---

## Basic Workflow Structure

### Workflow File Location

Create workflow files in `.github/workflows/` directory:

```
.github/
└── workflows/
    ├── train-model.yml
    ├── deploy-staging.yml
    └── deploy-production.yml
```

---

## Example 1: Model Training and Testing

```yaml
# .github/workflows/train-model.yml
name: Train and Test Model

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run weekly on Sundays at 00:00 UTC
    - cron: '0 0 * * 0'

jobs:
  train-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Generate training data
      run: |
        python generate_data.py
    
    - name: Train model
      run: |
        python train.py
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: trained-model
        path: models/
    
    - name: Upload coverage report
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

---

## Example 2: Deploy to Azure ML

```yaml
# .github/workflows/deploy-azure.yml
name: Deploy to Azure ML

on:
  workflow_run:
    workflows: ["Train and Test Model"]
    types:
      - completed
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install Azure ML SDK
      run: |
        pip install azure-ai-ml azure-identity
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: trained-model
        path: models/
    
    - name: Deploy to Azure ML
      run: |
        python azure-deployment/ml-studio/deploy_to_azure_ml.py
      env:
        AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
        AZURE_RESOURCE_GROUP: ${{ secrets.AZURE_RESOURCE_GROUP }}
        AZURE_WORKSPACE_NAME: ${{ secrets.AZURE_WORKSPACE_NAME }}
    
    - name: Test endpoint
      run: |
        python scripts/test_endpoint.py
```

---

## Example 3: Deploy to AWS SageMaker

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS SageMaker

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'
      - 'inference.py'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install boto3 sagemaker
    
    - name: Package model
      run: |
        tar -czf model.tar.gz -C models/ .
    
    - name: Upload to S3
      run: |
        aws s3 cp model.tar.gz s3://${{ secrets.S3_BUCKET }}/models/
    
    - name: Deploy to SageMaker
      run: |
        python scripts/deploy_sagemaker.py
```

---

## Example 4: Docker Build and Push

```yaml
# .github/workflows/docker-build.yml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

---

## Example 5: Multi-Environment Deployment

```yaml
# .github/workflows/deploy-multi-env.yml
name: Multi-Environment Deployment

on:
  push:
    branches:
      - develop
      - staging
      - main

jobs:
  determine-environment:
    runs-on: ubuntu-latest
    outputs:
      environment: ${{ steps.set-env.outputs.environment }}
    steps:
      - id: set-env
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "environment=production" >> $GITHUB_OUTPUT
          elif [[ "${{ github.ref }}" == "refs/heads/staging" ]]; then
            echo "environment=staging" >> $GITHUB_OUTPUT
          else
            echo "environment=development" >> $GITHUB_OUTPUT
          fi
  
  deploy:
    needs: determine-environment
    runs-on: ubuntu-latest
    environment: ${{ needs.determine-environment.outputs.environment }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Deploy to ${{ needs.determine-environment.outputs.environment }}
      run: |
        echo "Deploying to ${{ needs.determine-environment.outputs.environment }}"
        # Add deployment commands here
```

---

## Secrets Management

### Required Secrets

Add these secrets in GitHub repository settings:

**Azure:**
- `AZURE_CREDENTIALS`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WORKSPACE_NAME`

**AWS:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`

**GCP:**
- `GCP_PROJECT_ID`
- `GCP_SA_KEY`

### Setting Secrets

```bash
# Using GitHub CLI
gh secret set AZURE_SUBSCRIPTION_ID --body "your-subscription-id"
gh secret set AWS_ACCESS_KEY_ID --body "your-access-key"
```

---

## Matrix Testing

Test across multiple Python versions and platforms:

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Run tests
      run: pytest tests/
```

---

## Caching Dependencies

Speed up workflows with caching:

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

---

## Notifications

Send deployment notifications to Slack:

```yaml
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'Deployment to production completed'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Best Practices

**Use environment protection rules** for production deployments with required reviewers.

**Implement approval gates** for critical deployments.

**Use matrix builds** to test across multiple environments.

**Cache dependencies** to speed up workflow execution.

**Use secrets** for sensitive information, never commit credentials.

**Implement rollback strategies** for failed deployments.

**Monitor workflow execution** and set up alerts for failures.

**Use reusable workflows** to avoid duplication.

---

## Complete Example: Full CI/CD Pipeline

```yaml
# .github/workflows/complete-pipeline.yml
name: Complete ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install flake8 pytest pytest-cov
        pip install -r requirements.txt
    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Run tests
      run: pytest tests/ --cov=. --cov-report=xml
  
  train-model:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Train model
      run: python train.py
    - name: Upload model
      uses: actions/upload-artifact@v3
      with:
        name: model
        path: models/
  
  deploy-staging:
    needs: train-model
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - uses: actions/checkout@v3
    - name: Download model
      uses: actions/download-artifact@v3
      with:
        name: model
        path: models/
    - name: Deploy to staging
      run: echo "Deploy to staging"
  
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v3
    - name: Download model
      uses: actions/download-artifact@v3
      with:
        name: model
        path: models/
    - name: Deploy to production
      run: echo "Deploy to production"
```

---

**Last Updated:** January 2026
