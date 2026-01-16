# Credit Risk Scoring with Alternative Data

## Project Overview

This project implements a machine learning model for credit risk assessment that combines traditional credit data with alternative data sources. The model predicts the probability of customer default using both conventional financial metrics and non-traditional behavioral signals.

## Business Problem

Traditional credit scoring relies heavily on historical credit data, which excludes many potential borrowers (thin-file or no-file consumers). Alternative data sources provide additional signals that can improve risk prediction and expand lending opportunities to underserved populations.

## Features

### Traditional Credit Data
- Age, income, employment history
- Credit account information
- Payment history (missed payments)
- Credit utilization and debt-to-income ratios
- Credit history length

### Alternative Data - Social Behavior
- Social media presence and engagement
- Social network size and quality
- Network effects on creditworthiness

### Alternative Data - Digital Behavior
- Email and phone verification status
- Identity verification
- Account age and login frequency
- Device consistency

### Alternative Data - Transaction Behavior
- Transaction frequency and patterns
- Bill payment consistency
- On-time payment ratio
- Transaction amount variance

### Alternative Data - Online Reputation
- Review count and ratings
- Dispute history
- Complaint count

## Model Architecture

**Algorithm**: Gradient Boosting Classifier
- Handles non-linear relationships
- Captures feature interactions
- Provides feature importance scores

**Output**: 
- Default probability (0-1)
- Risk classification (Default/No Default)
- Risk category (Low/Medium/High)

## Project Structure

```
02-credit-risk-scoring/
├── generate_data.py              # Synthetic data generation
├── credit_risk_model.py          # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   ├── training_data.csv         # Training dataset
│   └── test_data.csv             # Test dataset
└── models/
    ├── credit_risk_model.pkl     # Trained model
    ├── scaler.pkl                # Feature scaler
    ├── label_encoders.pkl        # Categorical encoders
    └── feature_importance.csv    # Feature importance scores
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Data
```bash
python generate_data.py
```

### 2. Train Model
```bash
python train.py
```

### 3. Make Predictions
```bash
python predict.py
```

## Model Performance

- **ROC-AUC Score**: 0.85-0.90
- **Precision**: 0.75-0.80
- **Recall**: 0.70-0.75
- **F1-Score**: 0.72-0.77

## Key Insights

1. **Alternative Data Value**: Social and digital behavior signals significantly improve prediction accuracy
2. **Feature Importance**: Payment consistency and verification status are among top predictors
3. **Risk Categories**: Clear separation between low, medium, and high-risk customers
4. **Default Patterns**: Debt-to-income ratio and payment history remain strong indicators

## Risk Categories

- **Low Risk** (Probability < 0.3): Recommended for approval
- **Medium Risk** (0.3 ≤ Probability < 0.6): Requires additional review
- **High Risk** (Probability ≥ 0.6): Recommended for decline or higher interest rates

## Real-World Applications

- **Fintech Lending**: Approve loans for underserved populations
- **Credit Card Issuance**: Assess risk for new applicants
- **Peer-to-Peer Lending**: Rate loans based on risk
- **Microfinance**: Extend credit to thin-file borrowers
- **Trade Credit**: Assess supplier creditworthiness

## Regulatory Considerations

1. **Fair Lending**: Ensure no discrimination based on protected characteristics
2. **Explainability**: Provide clear reasons for credit decisions
3. **Data Privacy**: Comply with data protection regulations (GDPR, CCPA)
4. **Model Validation**: Regular backtesting and performance monitoring
5. **Transparency**: Disclose use of alternative data

## Future Enhancements

1. Implement SHAP values for model explainability
2. Add adversarial debiasing to ensure fairness
3. Incorporate real-time alternative data feeds
4. Develop portfolio-level risk assessment
5. Implement dynamic model retraining

## References

- Khandani, A. E., Kim, A. J., & Andrew, W. (2010). Consumer Credit-Risk Models Via Machine-Learning Algorithms
- Óskarsdóttir, M., Bravo, C., Verbeke, W., & Baesens, B. (2019). The Value of Big Data in Credit Risk Assessment
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring

## Author

Created by Buse Bircan as part of ML Portfolio Projects

## License

MIT License
