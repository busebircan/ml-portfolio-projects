# Call Drop Prediction

## Project Overview

This project predicts call drops in telecom networks before they occur, enabling proactive network maintenance and issue resolution. By identifying network conditions likely to cause drops, operators can take preventive action.

## Business Problem

Call drops negatively impact customer experience and lead to:
- Customer dissatisfaction
- Churn
- Support costs
- Revenue loss
- Regulatory issues

Predicting drops enables:
- Proactive network maintenance
- Resource allocation
- Early issue detection
- Improved service quality

## Features

### Network Conditions
- Signal strength, SNR, bandwidth, latency, jitter, packet loss

### Cell Tower Information
- Tower load, active users, age, maintenance history

### Call Characteristics
- Duration, type (voice/video/VoLTE), time of day, day of week

### Device Information
- Age, battery level, temperature, memory usage

### Movement and Location
- Speed, handover count, distance from tower, location type

### Historical Metrics
- Recent call drops, signal strength trends, network errors

## Model Architecture

**Algorithm**: Gradient Boosting Classifier
- Real-time prediction capability
- Feature importance for diagnostics
- Handles imbalanced data

## Project Structure

```
07-call-drop-prediction/
├── call_drop_model.py            # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    ├── drop_model.pkl            # Trained model
    ├── scaler.pkl                # Feature scaler
    └── feature_importance.csv    # Feature importance
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model
```bash
python train.py
```

### 2. Make Predictions
```bash
python predict.py
```

## Model Performance

- **ROC-AUC Score**: 0.82-0.87
- **Precision**: 0.70-0.75
- **Recall**: 0.75-0.80
- **F1-Score**: 0.72-0.77

## Top Call Drop Indicators

1. Signal strength (dBm)
2. Signal-to-noise ratio
3. Packet loss percentage
4. Tower load percentage
5. Latency
6. Call type (video more prone)
7. Handover count
8. Device temperature
9. Active users on tower
10. Network errors in last hour

## Risk Levels

- **Low Risk** (< 0.2): Normal operation
- **Medium Risk** (0.2-0.5): Monitor conditions
- **High Risk** (> 0.5): Preventive action needed

## Preventive Actions

### For High-Risk Calls
- Route to alternative tower
- Reduce call quality (video to voice)
- Alert network operations
- Trigger maintenance

### For Medium-Risk Calls
- Monitor in real-time
- Prepare backup resources
- Alert field teams

### For Low-Risk Calls
- Normal operation
- Continue monitoring

## Real-World Applications

- **Telecom**: Network quality management
- **Mobile operators**: Call quality assurance
- **5G networks**: Performance optimization
- **IoT networks**: Connection reliability
- **Emergency services**: Critical call reliability

## Benefits

1. **Improved quality**: Prevent calls from dropping
2. **Better experience**: Fewer customer complaints
3. **Cost savings**: Proactive maintenance cheaper than reactive
4. **Reduced churn**: Better service retention
5. **Network optimization**: Data-driven maintenance

## Network Conditions Impact

| Condition | Impact on Drop Rate |
|-----------|-------------------|
| Signal < -100 dBm | Very High |
| SNR < 5 dB | High |
| Packet Loss > 3% | High |
| Latency > 100 ms | Medium |
| Tower Load > 80% | Medium |

## Future Enhancements

1. Real-time streaming predictions
2. Anomaly detection for network issues
3. Root cause analysis
4. Predictive maintenance scheduling
5. Multi-cell optimization

## References

- 3GPP Standards for Network Quality
- Holma, H., & Toskala, A. (2011). LTE for 4G Mobile Broadband
- Sesia, S., Toufik, I., & Baker, M. (2009). LTE - The UMTS Long Term Evolution

## Author

Created by Buse Bircan as part of ML Portfolio Projects

## License

MIT License
