# ML Portfolio Projects

A comprehensive collection of 10 industry-specific machine learning projects showcasing real-world applications across Finance, Retail, Telecommunications, and Healthcare sectors.

## 📊 Project Overview

This repository contains production-ready machine learning projects with complete implementations, documentation, and synthetic datasets. Each project demonstrates best practices in data science, model development, and deployment.

## 📁 Repository Structure

```
ml-portfolio-projects/
├── Finance/
│   ├── 01-dynamic-pricing/
│   └── 02-credit-risk-scoring/
├── Retail & E-commerce/
│   ├── 03-demand-forecasting/
│   ├── 04-price-elasticity/
│   └── 05-multi-store-sku-forecasting/
├── Telecommunications/
│   ├── 06-network-churn-prediction/
│   └── 07-call-drop-prediction/
└── Healthcare/
    ├── 08-hospital-readmission/
    ├── 09-sepsis-detection/
    └── 10-drug-recommendation/
```

---

## 🏦 Finance

### 1. [Dynamic Pricing Optimization](Finance/01-dynamic-pricing/)
Optimize pricing strategies based on demand, competition, and inventory levels using machine learning.

**Key Features:**
- Real-time price optimization
- Demand prediction
- Inventory-aware pricing
- Profit maximization

**Technologies:** Gradient Boosting, Feature Engineering, Time Series Analysis  
**Model Performance:** R² = 0.85-0.90

### 2. [Credit Risk Scoring with Alternative Data](Finance/02-credit-risk-scoring/)
Assess credit risk using both traditional and alternative data sources for better lending decisions.

**Key Features:**
- Traditional credit metrics
- Alternative data integration (social, behavioral, digital)
- Risk categorization
- Fair lending compliance

**Technologies:** Gradient Boosting Classification, Feature Engineering  
**Model Performance:** ROC-AUC = 0.85-0.90

---

## 🛒 Retail & E-commerce

### 3. [Demand Forecasting with Inventory Optimization](Retail%20%26%20E-commerce/03-demand-forecasting/)
Forecast demand and optimize inventory parameters including safety stock and reorder points.

**Key Features:**
- Time-series demand forecasting
- Safety stock calculation
- Economic Order Quantity (EOQ)
- Multi-product support

**Technologies:** Gradient Boosting, Time Series Analysis, Inventory Theory  
**Model Performance:** MAPE = 10-15%

### 4. [Price Elasticity Analysis](Retail%20%26%20E-commerce/04-price-elasticity/)
Analyze how price changes affect demand across product categories.

**Key Features:**
- Category-specific elasticity
- Price scenario simulation
- Revenue impact analysis
- Competitive pricing insights

**Technologies:** Linear Regression, Random Forest, Econometric Analysis  
**Model Performance:** R² = 0.80-0.85

### 5. [Multi-Store SKU Forecasting](Retail%20%26%20E-commerce/05-multi-store-sku-forecasting/)
Forecast sales for multiple products across multiple store locations.

**Key Features:**
- Store-level modeling
- SKU-specific patterns
- Temporal dynamics
- Scalable architecture

**Technologies:** Gradient Boosting (per-store models), Feature Engineering  
**Model Performance:** MAPE = 10-15%

---

## 📱 Telecommunications

### 6. [Network Churn Prediction](Telecommunications/06-network-churn-prediction/)
Predict which customers will switch carriers based on usage patterns and service quality.

**Key Features:**
- Usage pattern analysis
- Service quality metrics
- Competitive factor assessment
- Risk stratification

**Technologies:** Gradient Boosting Classification, Feature Importance  
**Model Performance:** ROC-AUC = 0.85-0.90

### 7. [Call Drop Prediction](Telecommunications/07-call-drop-prediction/)
Identify network issues before customers complain by predicting call drops.

**Key Features:**
- Real-time prediction
- Network condition monitoring
- Proactive maintenance
- Risk level categorization

**Technologies:** Gradient Boosting Classification, Real-time Processing  
**Model Performance:** ROC-AUC = 0.82-0.87

---

## 🏥 Healthcare

### 8. [Hospital Readmission Prediction](Healthcare/08-hospital-readmission/)
Predict 30-day readmissions for diabetic patients using EHR data.

**Key Features:**
- Diabetes-specific risk factors
- Comorbidity assessment
- Social factor integration
- Intervention targeting

**Technologies:** Gradient Boosting Classification, Feature Engineering  
**Model Performance:** ROC-AUC = 0.80-0.85

### 9. [Early Sepsis Detection](Healthcare/09-sepsis-detection/)
Predict sepsis 6 hours before clinical onset using time-series clinical data.

**Key Features:**
- Early warning system
- Vital sign trend analysis
- Lab value integration
- Real-time monitoring

**Technologies:** Gradient Boosting Classification, Time Series Analysis  
**Model Performance:** ROC-AUC = 0.85-0.90

### 10. [Drug Recommendation System](Healthcare/10-drug-recommendation/)
Recommend medications based on patient conditions, lab values, and medical history.

**Key Features:**
- Multi-label drug recommendations
- Contraindication checking
- Allergy awareness
- Evidence-based suggestions

**Technologies:** Multi-label Random Forest, Medical Knowledge Integration  
**Model Performance:** Accuracy = 0.80-0.90

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/busebircan/ml-portfolio-projects.git
cd ml-portfolio-projects
```

2. Navigate to any project folder:
```bash
cd Finance/01-dynamic-pricing
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running a Project

Each project follows the same structure:

```bash
# Train the model
python train.py

# Make predictions
python predict.py
```

---

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn**: Core ML algorithms
- **Gradient Boosting**: Primary modeling technique
- **Random Forest**: Ensemble methods
- **Linear Regression**: Econometric analysis

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Model Persistence
- **joblib**: Model serialization

---

## 📊 Model Comparison

| Project | Algorithm | Task Type | Performance |
|---------|-----------|-----------|-------------|
| Dynamic Pricing | Gradient Boosting | Regression | R² = 0.87 |
| Credit Risk | Gradient Boosting | Classification | ROC-AUC = 0.88 |
| Demand Forecast | Gradient Boosting | Regression | MAPE = 12% |
| Price Elasticity | Random Forest | Regression | R² = 0.82 |
| Multi-Store SKU | Gradient Boosting | Regression | MAPE = 12% |
| Churn Prediction | Gradient Boosting | Classification | ROC-AUC = 0.87 |
| Call Drop | Gradient Boosting | Classification | ROC-AUC = 0.85 |
| Readmission | Gradient Boosting | Classification | ROC-AUC = 0.82 |
| Sepsis Detection | Gradient Boosting | Classification | ROC-AUC = 0.88 |
| Drug Recommendation | Random Forest | Multi-label | Accuracy = 0.85 |

---

## 🎯 Key Features

- **Production-Ready Code**: Clean, well-documented, and modular
- **Synthetic Datasets**: Realistic data generation for each domain
- **Comprehensive Documentation**: Detailed README for each project
- **Best Practices**: Feature engineering, model evaluation, validation
- **Scalability**: Designed for real-world deployment
- **Reproducibility**: Fixed random seeds for consistent results

---

## 📚 Learning Outcomes

By exploring these projects, you'll learn:

1. **Data Science Workflow**: From data generation to model deployment
2. **Domain Knowledge**: Applications across multiple industries
3. **Feature Engineering**: Creating meaningful features for ML models
4. **Model Selection**: Choosing appropriate algorithms for different tasks
5. **Evaluation Metrics**: Proper model assessment techniques
6. **Best Practices**: Production-ready code standards

---

## 🔍 Project Details

Each project includes:

- **README.md**: Comprehensive project documentation
- **model.py**: Core ML implementation
- **train.py**: Model training script
- **predict.py**: Prediction and inference script
- **requirements.txt**: Python dependencies
- **models/**: Trained model artifacts (generated after training)

---

## 💡 Use Cases

### Finance
- Real-time pricing optimization for e-commerce
- Credit approval decisions for fintech platforms
- Risk assessment for lending decisions

### Retail
- Inventory optimization for supply chains
- Price optimization across categories
- Multi-location demand planning

### Telecommunications
- Customer retention programs
- Network quality management
- Proactive maintenance

### Healthcare
- Hospital readmission prevention
- Sepsis early warning systems
- Clinical decision support

---

## 🤝 Contributing

Feel free to fork, modify, and extend these projects for your own use cases. Each project is designed to be modular and adaptable.

---

## 📖 References

Each project includes relevant academic papers and industry resources in its README.

---

## 📝 License

MIT License - Feel free to use these projects for learning and commercial purposes.

---

## 👤 Author

Created by Buse Bircan as a comprehensive ML portfolio showcasing real-world applications.

---

## 🔗 Quick Navigation

### Finance
- [Dynamic Pricing Optimization](Finance/01-dynamic-pricing/README.md)
- [Credit Risk Scoring](Finance/02-credit-risk-scoring/README.md)

### Retail & E-commerce
- [Demand Forecasting](Retail%20%26%20E-commerce/03-demand-forecasting/README.md)
- [Price Elasticity Analysis](Retail%20%26%20E-commerce/04-price-elasticity/README.md)
- [Multi-Store SKU Forecasting](Retail%20%26%20E-commerce/05-multi-store-sku-forecasting/README.md)

### Telecommunications
- [Network Churn Prediction](Telecommunications/06-network-churn-prediction/README.md)
- [Call Drop Prediction](Telecommunications/07-call-drop-prediction/README.md)

### Healthcare
- [Hospital Readmission Prediction](Healthcare/08-hospital-readmission/README.md)
- [Early Sepsis Detection](Healthcare/09-sepsis-detection/README.md)
- [Drug Recommendation System](Healthcare/10-drug-recommendation/README.md)

---

**Last Updated:** January 2026

**Status:** Complete with 10 production-ready projects organized by industry
