# Network Churn Prediction

## Project Overview

This project predicts which telecom customers will switch to a competitor based on their usage patterns, service satisfaction, and account characteristics. The model enables proactive retention strategies.

## Business Problem

Customer churn is costly for telecom companies. Predicting churn allows:
- Proactive retention campaigns
- Targeted incentives for at-risk customers
- Resource optimization
- Revenue protection
- Customer lifetime value maximization

## Features

### Customer Demographics
- Age, tenure, contract type, internet service

### Usage Patterns
- Monthly minutes, data usage, call frequency, SMS usage

### Service Quality
- Call quality score, network satisfaction, network performance metrics

### Account Information
- Monthly charges, total charges, payment method, billing preferences

### Customer Service
- Support calls, complaints, tech support usage, interaction history

### Network Performance
- Latency, packet loss, dropped calls, outage incidents

### Competitive Factors
- Competitor offers, price comparison, market competition

## Model Architecture

**Algorithm**: Gradient Boosting Classifier
- Captures complex churn patterns
- Provides feature importance scores
- Handles imbalanced data

## Project Structure

```
06-network-churn-prediction/
├── churn_prediction_model.py     # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    ├── churn_model.pkl           # Trained model
    ├── scaler.pkl                # Feature scaler
    ├── label_encoders.pkl        # Categorical encoders
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

- **ROC-AUC Score**: 0.85-0.90
- **Precision**: 0.75-0.80
- **Recall**: 0.70-0.75
- **F1-Score**: 0.72-0.77

## Top Churn Indicators

1. Network satisfaction score
2. Complaint count
3. Dropped calls per month
4. Price compared to market
5. Competitor offers received
6. Support calls
7. Contract type (month-to-month higher risk)
8. Tenure (newer customers higher risk)

## Churn Risk Levels

- **Low Risk** (< 0.3): Stable customers
- **Medium Risk** (0.3-0.6): Requires monitoring
- **High Risk** (> 0.6): Immediate retention action needed

## Retention Strategies

### For High-Risk Customers
- Personalized retention offers
- Service quality improvements
- Loyalty rewards
- Direct outreach

### For Medium-Risk Customers
- Proactive support
- Usage-based recommendations
- Upgrade incentives

### For Low-Risk Customers
- Maintain service quality
- Regular engagement
- Upsell opportunities

## Real-World Applications

- **Telecom**: Customer retention programs
- **ISP**: Internet service provider churn prediction
- **Mobile**: Carrier churn prediction
- **Cable**: Cable TV subscriber retention
- **Utilities**: Customer retention

## Benefits

1. **Reduced churn**: Identify at-risk customers early
2. **Cost savings**: Targeted retention is cheaper than acquisition
3. **Revenue protection**: Maintain customer base
4. **Improved satisfaction**: Address issues proactively
5. **Better ROI**: Focus resources on high-value customers

## Future Enhancements

1. Survival analysis for time-to-churn prediction
2. Causal inference for retention strategies
3. Real-time churn scoring
4. Personalized retention recommendations
5. A/B testing framework for retention tactics

## References

- Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Sun, B. (2006). Defection Detection
- Verbeke, W., Martens, D., Mues, C., & Baesens, B. (2011). Building Comprehensible Customer Churn Prediction Models
- Burez, J., & Van den Poel, D. (2007). CRM at a Pay-TV Company

## Author

Created as part of ML Portfolio Projects

## License

MIT License
