# Model Performance Monitoring

Comprehensive guide for monitoring ML models in production to detect performance degradation, data drift, and concept drift.

## Overview

Production ML models require continuous monitoring to ensure they maintain expected performance levels. This guide covers key metrics, tools, and strategies for effective model monitoring.

---

## Key Monitoring Metrics

### 1. Model Performance Metrics

**Classification Models:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion matrix
- Class distribution

**Regression Models:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Prediction distribution

### 2. Data Quality Metrics

- **Missing values:** Percentage of null/missing features
- **Feature distributions:** Statistical properties (mean, std, min, max)
- **Outliers:** Values outside expected ranges
- **Data types:** Unexpected type changes

### 3. System Performance Metrics

- **Latency:** Prediction response time (p50, p95, p99)
- **Throughput:** Requests per second
- **Error rate:** Failed predictions / total predictions
- **Resource utilization:** CPU, memory, GPU usage

### 4. Business Metrics

- **Prediction distribution:** Are predictions reasonable?
- **User engagement:** Click-through rates, conversions
- **Revenue impact:** Financial outcomes
- **Customer satisfaction:** Feedback scores

---

## Data Drift Detection

Data drift occurs when the statistical properties of input features change over time.

### Statistical Tests

```python
import numpy as np
from scipy import stats

def detect_drift_ks_test(reference_data, current_data, threshold=0.05):
    """
    Kolmogorov-Smirnov test for drift detection
    
    Args:
        reference_data: Training/baseline data
        current_data: Current production data
        threshold: p-value threshold
    
    Returns:
        bool: True if drift detected
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    return p_value < threshold

# Example usage
baseline_feature = training_data['price']
current_feature = production_data['price']

if detect_drift_ks_test(baseline_feature, current_feature):
    print("⚠️ Data drift detected in 'price' feature")
```

### Population Stability Index (PSI)

```python
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change
    """
    # Create bins
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    
    # Calculate distributions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi

# Example
psi_score = calculate_psi(training_data['demand'], production_data['demand'])
print(f"PSI Score: {psi_score:.4f}")

if psi_score > 0.2:
    print("⚠️ Significant population shift detected")
```

---

## Concept Drift Detection

Concept drift occurs when the relationship between features and target changes.

### Performance-Based Detection

```python
import pandas as pd
from datetime import datetime, timedelta

def monitor_performance_drift(predictions_log, window_days=7, threshold=0.05):
    """
    Monitor model performance over time
    
    Args:
        predictions_log: DataFrame with columns [timestamp, prediction, actual, metric]
        window_days: Rolling window size
        threshold: Acceptable performance degradation
    """
    # Calculate rolling performance
    predictions_log['date'] = pd.to_datetime(predictions_log['timestamp']).dt.date
    
    daily_performance = predictions_log.groupby('date').apply(
        lambda x: calculate_metric(x['actual'], x['prediction'])
    )
    
    # Calculate rolling mean
    rolling_performance = daily_performance.rolling(window=window_days).mean()
    
    # Detect drift
    baseline_performance = rolling_performance.iloc[0]
    current_performance = rolling_performance.iloc[-1]
    
    degradation = (baseline_performance - current_performance) / baseline_performance
    
    if degradation > threshold:
        return True, degradation
    
    return False, degradation

def calculate_metric(actual, predicted):
    """Calculate appropriate metric (e.g., accuracy, RMSE)"""
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(actual, predicted, squared=False)
```

---

## Monitoring Implementation

### 1. Logging Predictions

```python
import json
import logging
from datetime import datetime

class PredictionLogger:
    """Log predictions for monitoring"""
    
    def __init__(self, log_file='predictions.log'):
        self.logger = logging.getLogger('predictions')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_prediction(self, features, prediction, confidence=None, metadata=None):
        """Log a single prediction"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': float(prediction),
            'confidence': float(confidence) if confidence else None,
            'metadata': metadata or {}
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_feedback(self, prediction_id, actual_value):
        """Log actual outcome for comparison"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction_id': prediction_id,
            'actual': float(actual_value),
            'type': 'feedback'
        }
        
        self.logger.info(json.dumps(feedback_entry))

# Usage
logger = PredictionLogger()

# Log prediction
logger.log_prediction(
    features={'price': 100, 'demand': 50},
    prediction=105.5,
    confidence=0.85,
    metadata={'model_version': 'v1.0'}
)

# Log actual outcome later
logger.log_feedback(prediction_id='pred_123', actual_value=107.2)
```

### 2. Real-Time Monitoring Dashboard

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_monitoring_dashboard(predictions_df):
    """Create interactive monitoring dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Prediction Distribution', 'Latency Over Time',
                       'Error Rate', 'Feature Drift')
    )
    
    # Prediction distribution
    fig.add_trace(
        go.Histogram(x=predictions_df['prediction'], name='Predictions'),
        row=1, col=1
    )
    
    # Latency over time
    fig.add_trace(
        go.Scatter(x=predictions_df['timestamp'], y=predictions_df['latency'],
                  mode='lines', name='Latency'),
        row=1, col=2
    )
    
    # Error rate
    error_rate = predictions_df.groupby(pd.Grouper(key='timestamp', freq='1H'))['error'].mean()
    fig.add_trace(
        go.Scatter(x=error_rate.index, y=error_rate.values,
                  mode='lines', name='Error Rate'),
        row=2, col=1
    )
    
    # Feature drift (PSI)
    fig.add_trace(
        go.Bar(x=predictions_df['feature_name'], y=predictions_df['psi'],
              name='PSI Score'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Model Monitoring Dashboard")
    
    return fig
```

### 3. Alerting System

```python
import smtplib
from email.mime.text import MIMEText

class ModelAlertSystem:
    """Send alerts when issues are detected"""
    
    def __init__(self, smtp_server, sender_email, recipient_emails):
        self.smtp_server = smtp_server
        self.sender_email = sender_email
        self.recipient_emails = recipient_emails
    
    def send_alert(self, alert_type, message, severity='WARNING'):
        """Send email alert"""
        subject = f"[{severity}] Model Alert: {alert_type}"
        
        body = f"""
        Alert Type: {alert_type}
        Severity: {severity}
        Timestamp: {datetime.now().isoformat()}
        
        Details:
        {message}
        
        Please investigate and take appropriate action.
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.recipient_emails)
        
        with smtplib.SMTP(self.smtp_server) as server:
            server.send_message(msg)
    
    def check_and_alert(self, metrics):
        """Check metrics and send alerts if thresholds exceeded"""
        
        # Check latency
        if metrics['latency_p95'] > 1000:  # 1 second
            self.send_alert(
                'High Latency',
                f"P95 latency: {metrics['latency_p95']}ms",
                severity='WARNING'
            )
        
        # Check error rate
        if metrics['error_rate'] > 0.05:  # 5%
            self.send_alert(
                'High Error Rate',
                f"Error rate: {metrics['error_rate']:.2%}",
                severity='CRITICAL'
            )
        
        # Check data drift
        if metrics['max_psi'] > 0.2:
            self.send_alert(
                'Data Drift Detected',
                f"Maximum PSI: {metrics['max_psi']:.4f}",
                severity='WARNING'
            )
        
        # Check performance degradation
        if metrics['performance_drop'] > 0.1:  # 10%
            self.send_alert(
                'Performance Degradation',
                f"Performance drop: {metrics['performance_drop']:.2%}",
                severity='CRITICAL'
            )
```

---

## Monitoring Tools

### Application Insights (Azure)

```python
from applicationinsights import TelemetryClient

tc = TelemetryClient('your-instrumentation-key')

# Track prediction
tc.track_event('prediction', {
    'model_version': 'v1.0',
    'prediction': 105.5,
    'confidence': 0.85
})

# Track metric
tc.track_metric('prediction_latency', 45.2)

# Track exception
try:
    prediction = model.predict(features)
except Exception as e:
    tc.track_exception()

tc.flush()
```

### CloudWatch (AWS)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Put custom metric
cloudwatch.put_metric_data(
    Namespace='MLModel',
    MetricData=[
        {
            'MetricName': 'PredictionLatency',
            'Value': 45.2,
            'Unit': 'Milliseconds',
            'Timestamp': datetime.now()
        }
    ]
)
```

### Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

# Record metrics
@prediction_latency.time()
def make_prediction(features):
    prediction_counter.inc()
    result = model.predict(features)
    return result
```

---

## Retraining Triggers

Automate model retraining based on monitoring signals:

```python
class RetrainingManager:
    """Manage model retraining decisions"""
    
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.retrain_triggered = False
    
    def should_retrain(self, metrics):
        """Determine if retraining is needed"""
        
        reasons = []
        
        # Check performance degradation
        if metrics['performance_drop'] > self.thresholds['performance']:
            reasons.append(f"Performance degraded by {metrics['performance_drop']:.2%}")
        
        # Check data drift
        if metrics['max_psi'] > self.thresholds['psi']:
            reasons.append(f"Data drift detected (PSI: {metrics['max_psi']:.4f})")
        
        # Check time since last training
        days_since_training = (datetime.now() - metrics['last_training_date']).days
        if days_since_training > self.thresholds['max_days']:
            reasons.append(f"Model age: {days_since_training} days")
        
        # Check data volume
        if metrics['new_data_points'] > self.thresholds['min_data_points']:
            reasons.append(f"Sufficient new data: {metrics['new_data_points']} points")
        
        if reasons:
            self.retrain_triggered = True
            return True, reasons
        
        return False, []
    
    def trigger_retraining(self, reasons):
        """Initiate retraining pipeline"""
        print("🔄 Triggering model retraining")
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
        
        # Trigger retraining pipeline (e.g., via API, message queue)
        # trigger_training_pipeline()

# Usage
manager = RetrainingManager(thresholds={
    'performance': 0.1,  # 10% degradation
    'psi': 0.2,
    'max_days': 90,
    'min_data_points': 10000
})

metrics = {
    'performance_drop': 0.12,
    'max_psi': 0.25,
    'last_training_date': datetime(2024, 1, 1),
    'new_data_points': 15000
}

should_retrain, reasons = manager.should_retrain(metrics)
if should_retrain:
    manager.trigger_retraining(reasons)
```

---

## Best Practices

**Monitor continuously,** not just at deployment. Set up automated monitoring pipelines.

**Define clear thresholds** for alerts based on business requirements.

**Log all predictions** with timestamps for retrospective analysis.

**Collect ground truth** whenever possible to measure actual performance.

**Set up dashboards** for real-time visibility into model health.

**Automate retraining** based on monitoring signals.

**Version everything:** models, data, code, and configurations.

**Test monitoring systems** regularly to ensure they work when needed.

**Document baseline metrics** from training for comparison.

**Involve stakeholders** in defining acceptable performance levels.

---

## Monitoring Checklist

- [ ] Prediction logging implemented
- [ ] Performance metrics tracked
- [ ] Data drift detection configured
- [ ] Concept drift monitoring active
- [ ] Alerting system set up
- [ ] Dashboards created
- [ ] Retraining triggers defined
- [ ] Ground truth collection process
- [ ] Baseline metrics documented
- [ ] Stakeholder thresholds agreed

---

**Last Updated:** January 2026
