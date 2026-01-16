# Security Best Practices for ML Deployments

Comprehensive security guide for protecting ML models, data, and infrastructure in production environments.

## Overview

ML systems face unique security challenges including model theft, adversarial attacks, data poisoning, and privacy violations. This guide covers essential security practices for production ML deployments.

---

## Authentication and Authorization

### API Key Authentication

```python
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
import secrets

app = FastAPI()

API_KEY = "your-secret-api-key"  # Store in environment variable
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key"""
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key"
        )
    return api_key

@app.post("/predict")
async def predict(data: dict, api_key: str = Security(verify_api_key)):
    # Your prediction logic
    return {"prediction": 123}
```

### OAuth2 Authentication

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

app = FastAPI()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate token and get user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

@app.post("/predict")
async def predict(data: dict, current_user: str = Depends(get_current_user)):
    # Your prediction logic
    return {"prediction": 123, "user": current_user}
```

---

## Input Validation

### Schema Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List

class PredictionInput(BaseModel):
    """Validated input schema"""
    price: float = Field(..., gt=0, lt=10000, description="Price must be positive")
    demand: int = Field(..., ge=0, le=1000, description="Demand between 0-1000")
    inventory: int = Field(..., ge=0, description="Inventory must be non-negative")
    
    @validator('price')
    def validate_price(cls, v):
        """Custom price validation"""
        if v < 10 or v > 5000:
            raise ValueError('Price must be between 10 and 5000')
        return v
    
    @validator('demand')
    def validate_demand(cls, v):
        """Custom demand validation"""
        if v < 0:
            raise ValueError('Demand cannot be negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "price": 100.0,
                "demand": 50,
                "inventory": 200
            }
        }
```

### Sanitization

```python
import re
from typing import Any, Dict

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data"""
    sanitized = {}
    
    for key, value in data.items():
        # Remove special characters from string keys
        clean_key = re.sub(r'[^a-zA-Z0-9_]', '', key)
        
        # Sanitize string values
        if isinstance(value, str):
            # Remove potential SQL injection patterns
            value = re.sub(r'[;\'"\\]', '', value)
            # Limit length
            value = value[:1000]
        
        # Validate numeric ranges
        elif isinstance(value, (int, float)):
            # Clip to reasonable ranges
            value = max(-1e10, min(1e10, value))
        
        sanitized[clean_key] = value
    
    return sanitized
```

---

## Rate Limiting

### Simple Rate Limiter

```python
from fastapi import FastAPI, Request, HTTPException
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

app = FastAPI()

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    response = await call_next(request)
    return response
```

---

## Model Security

### Model Encryption

```python
from cryptography.fernet import Fernet
import joblib
import io

class EncryptedModelStorage:
    """Encrypt and decrypt model files"""
    
    def __init__(self, key=None):
        """Initialize with encryption key"""
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self.key = key
    
    def save_encrypted_model(self, model, filepath):
        """Save model in encrypted format"""
        # Serialize model
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()
        
        # Encrypt
        encrypted_data = self.cipher.encrypt(model_bytes)
        
        # Save
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)
    
    def load_encrypted_model(self, filepath):
        """Load encrypted model"""
        # Read encrypted data
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        model_bytes = self.cipher.decrypt(encrypted_data)
        
        # Deserialize
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)
        
        return model

# Usage
storage = EncryptedModelStorage()

# Save encrypted
storage.save_encrypted_model(model, 'model_encrypted.pkl')

# Load encrypted
model = storage.load_encrypted_model('model_encrypted.pkl')
```

### Model Watermarking

```python
import numpy as np

def add_watermark(model, watermark_data, watermark_labels):
    """
    Add watermark to model for ownership verification
    
    Args:
        model: Trained model
        watermark_data: Special data points
        watermark_labels: Expected predictions
    """
    # Fine-tune model on watermark data
    model.fit(watermark_data, watermark_labels)
    return model

def verify_watermark(model, watermark_data, watermark_labels, threshold=0.9):
    """Verify model ownership via watermark"""
    predictions = model.predict(watermark_data)
    accuracy = np.mean(predictions == watermark_labels)
    return accuracy > threshold
```

---

## Data Privacy

### Differential Privacy

```python
import numpy as np

def add_laplace_noise(data, epsilon=1.0, sensitivity=1.0):
    """
    Add Laplace noise for differential privacy
    
    Args:
        data: Original data
        epsilon: Privacy budget (lower = more privacy)
        sensitivity: Maximum change in output
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# Example
private_predictions = add_laplace_noise(predictions, epsilon=0.5)
```

### Data Anonymization

```python
import hashlib

def anonymize_user_id(user_id: str, salt: str = "secret-salt") -> str:
    """Anonymize user IDs"""
    return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()

def remove_pii(data: dict) -> dict:
    """Remove personally identifiable information"""
    pii_fields = ['email', 'phone', 'ssn', 'address', 'name']
    
    cleaned_data = data.copy()
    for field in pii_fields:
        if field in cleaned_data:
            del cleaned_data[field]
    
    return cleaned_data
```

---

## Secure Communication

### HTTPS/TLS

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="/path/to/key.pem",
        ssl_certfile="/path/to/cert.pem"
    )
```

### Request Signing

```python
import hmac
import hashlib
from datetime import datetime

def sign_request(payload: str, secret_key: str) -> str:
    """Sign request with HMAC"""
    timestamp = str(int(datetime.now().timestamp()))
    message = f"{timestamp}.{payload}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{timestamp}.{signature}"

def verify_signature(payload: str, signature: str, secret_key: str, max_age=300) -> bool:
    """Verify request signature"""
    try:
        timestamp, sig = signature.split('.')
        
        # Check timestamp
        age = int(datetime.now().timestamp()) - int(timestamp)
        if age > max_age:
            return False
        
        # Verify signature
        expected_sig = sign_request(payload, secret_key).split('.')[1]
        return hmac.compare_digest(sig, expected_sig)
    
    except:
        return False
```

---

## Secrets Management

### Environment Variables

```python
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Access secrets
API_KEY = os.getenv('API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
SECRET_KEY = os.getenv('SECRET_KEY')

# Validate
if not API_KEY:
    raise ValueError("API_KEY not set")
```

### Azure Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def get_secret(vault_url: str, secret_name: str) -> str:
    """Retrieve secret from Azure Key Vault"""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    secret = client.get_secret(secret_name)
    return secret.value

# Usage
api_key = get_secret(
    vault_url="https://your-vault.vault.azure.net/",
    secret_name="api-key"
)
```

---

## Logging and Auditing

### Secure Logging

```python
import logging
import json
from datetime import datetime

class SecureLogger:
    """Logger that masks sensitive information"""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('secure.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def mask_sensitive_data(self, data: dict) -> dict:
        """Mask sensitive fields"""
        sensitive_fields = ['password', 'api_key', 'token', 'ssn', 'credit_card']
        
        masked_data = data.copy()
        for field in sensitive_fields:
            if field in masked_data:
                masked_data[field] = '***REDACTED***'
        
        return masked_data
    
    def log_request(self, user_id: str, endpoint: str, data: dict):
        """Log API request"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'endpoint': endpoint,
            'data': self.mask_sensitive_data(data)
        }
        self.logger.info(json.dumps(log_entry))

# Usage
logger = SecureLogger('api')
logger.log_request('user123', '/predict', {'price': 100, 'api_key': 'secret'})
```

---

## Security Checklist

### Authentication & Authorization
- [ ] API key or OAuth2 authentication implemented
- [ ] Role-based access control (RBAC) configured
- [ ] Token expiration and refresh implemented
- [ ] Multi-factor authentication (MFA) considered

### Input Security
- [ ] Input validation with Pydantic schemas
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] Rate limiting implemented
- [ ] Request size limits enforced

### Model Security
- [ ] Model files encrypted at rest
- [ ] Model watermarking implemented
- [ ] Model versioning and access control
- [ ] Adversarial robustness tested

### Data Privacy
- [ ] PII removal/anonymization
- [ ] Differential privacy considered
- [ ] Data encryption in transit and at rest
- [ ] GDPR/HIPAA compliance reviewed

### Infrastructure
- [ ] HTTPS/TLS enabled
- [ ] Firewall rules configured
- [ ] Network segmentation implemented
- [ ] Regular security updates applied

### Monitoring & Auditing
- [ ] Security logging enabled
- [ ] Audit trails maintained
- [ ] Anomaly detection configured
- [ ] Incident response plan documented

### Secrets Management
- [ ] No secrets in code or version control
- [ ] Environment variables or key vault used
- [ ] Secret rotation policy defined
- [ ] Access to secrets logged

---

## Incident Response

### Security Incident Plan

1. **Detection:** Monitor for security events
2. **Containment:** Isolate affected systems
3. **Investigation:** Analyze logs and identify root cause
4. **Remediation:** Fix vulnerabilities
5. **Recovery:** Restore normal operations
6. **Post-mortem:** Document lessons learned

---

## Compliance Frameworks

### GDPR (General Data Protection Regulation)
- Right to access
- Right to erasure
- Data portability
- Privacy by design

### HIPAA (Health Insurance Portability and Accountability Act)
- PHI protection
- Access controls
- Audit trails
- Encryption

### SOC 2
- Security
- Availability
- Processing integrity
- Confidentiality
- Privacy

---

**Last Updated:** January 2026
