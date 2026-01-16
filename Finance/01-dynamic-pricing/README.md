# Dynamic Pricing Optimization

## Project Overview

This project implements a machine learning-based dynamic pricing strategy that adjusts prices in real-time based on demand, competition, inventory levels, and customer segments. The model optimizes pricing to maximize revenue while considering market dynamics.

## Business Problem

Traditional static pricing strategies fail to capture market opportunities. Dynamic pricing allows businesses to:
- Maximize revenue by adjusting prices based on demand fluctuations
- Respond to competitor pricing strategies
- Optimize inventory levels by adjusting prices
- Segment customers and offer personalized pricing
- Reduce stockouts and overstock situations

## Features

### Input Features
- **Time-based**: Day of week, month, weekend indicator, holiday indicator
- **Demand factors**: Base demand, seasonal factors, demand shocks
- **Competition**: Competitor prices, competitor count, market share
- **Inventory**: Inventory level, inventory cost, stockout penalty
- **Customer**: Customer segment, price elasticity, customer lifetime value
- **Costs**: Production cost, distribution cost, marketing spend

### Model Components
1. **Price Prediction Model**: Gradient Boosting Regressor
   - Predicts optimal price based on market conditions
   - Considers all input features
   
2. **Demand Prediction Model**: Random Forest Regressor
   - Predicts demand at different price points
   - Captures non-linear relationships

3. **Optimization Engine**: 
   - Adjusts prices based on inventory levels
   - Maximizes profit while maintaining competitiveness
   - Ensures prices remain above production costs

## Project Structure

```
01-dynamic-pricing/
├── generate_data.py              # Synthetic data generation
├── dynamic_pricing_model.py      # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   ├── training_data.csv         # Training dataset
│   └── test_data.csv             # Test dataset
└── models/
    ├── price_model.pkl           # Trained price model
    ├── demand_model.pkl          # Trained demand model
    ├── scaler.pkl                # Feature scaler
    └── label_encoders.pkl        # Categorical encoders
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

### Price Prediction Model
- **RMSE**: ~3-4 (prediction error in price units)
- **R² Score**: 0.85-0.90
- **MAE**: ~2-3

### Demand Prediction Model
- **RMSE**: ~8-10 (prediction error in units)
- **R² Score**: 0.80-0.85
- **MAE**: ~5-7

## Key Insights

1. **Inventory Impact**: High inventory levels warrant price reductions to increase demand
2. **Competition Response**: Prices should be adjusted relative to competitor pricing
3. **Seasonal Patterns**: Demand varies significantly by season and day of week
4. **Customer Segmentation**: Premium customers show different price sensitivity
5. **Cost Constraints**: Prices must remain above production costs for profitability

## Real-World Applications

- **E-commerce**: Dynamic pricing for online retailers
- **Airlines**: Ticket price optimization based on demand
- **Hotels**: Room rate optimization based on occupancy
- **Ride-sharing**: Surge pricing based on demand
- **Retail**: Price optimization for physical stores

## Future Enhancements

1. Implement reinforcement learning for real-time optimization
2. Add A/B testing framework for price experimentation
3. Incorporate customer behavior predictions
4. Add multi-product pricing optimization
5. Implement price elasticity estimation

## References

- Talluri, K., & Van Ryzin, G. (2004). The Theory and Practice of Revenue Management
- Bitran, G., & Caldentey, R. (2003). An Overview of Pricing Models for Revenue Management
- Elmaghraby, W., & Keskinocak, P. (2003). Dynamic Pricing in the Presence of Inventory Considerations

## Author

Created as part of ML Portfolio Projects

## License

MIT License
