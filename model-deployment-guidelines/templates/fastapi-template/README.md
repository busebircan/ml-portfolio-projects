# FastAPI ML Model Template

Production-ready FastAPI template for deploying scikit-learn models as REST APIs.

## Features

- ✅ Single and batch predictions
- ✅ Automatic API documentation (Swagger/ReDoc)
- ✅ Input validation with Pydantic
- ✅ Health check endpoint
- ✅ CORS support
- ✅ Logging
- ✅ Error handling
- ✅ Model information endpoint

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Model

Place your trained model files in the `models/` directory:
- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler (optional)
- `feature_names.txt` - Feature names (optional)

### 3. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Access API Documentation

Open your browser and navigate to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 100.0,
      "feature2": 50,
      "feature3": 200
    }
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"feature1": 100.0, "feature2": 50, "feature3": 200},
      {"feature1": 110.0, "feature2": 55, "feature3": 210}
    ]
  }'
```

### Model Information
```bash
curl http://localhost:8000/model/info
```

## Customization

### 1. Update Feature Schema

Modify the `PredictionInput` class in `main.py` to match your model's features:

```python
class PredictionInput(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "your_feature_1": 100.0,
                    "your_feature_2": 50,
                    # Add your features here
                }
            }
        }
```

### 2. Add Custom Preprocessing

Modify the `preprocess_features` function to add custom preprocessing logic.

### 3. Add Authentication

Add API key authentication or OAuth2:

```python
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(input_data: PredictionInput, api_key: str = Depends(verify_api_key)):
    # Your prediction code
    pass
```

## Docker Deployment

Build and run as a Docker container:

```bash
docker build -t ml-api:latest .
docker run -p 8000:8000 ml-api:latest
```

## Production Considerations

- Set up proper logging (e.g., to file or external service)
- Implement rate limiting
- Add authentication and authorization
- Configure CORS appropriately
- Use environment variables for configuration
- Set up monitoring and alerting
- Implement caching for frequent predictions
- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)

## Running with Gunicorn

```bash
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
