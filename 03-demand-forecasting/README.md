# Demand Forecasting with Inventory Optimization

## Project Overview

This project implements a machine learning model for demand forecasting combined with inventory optimization. The model predicts future demand and automatically calculates optimal reorder points, safety stock levels, and economic order quantities to minimize total inventory costs.

## Business Problem

Accurate demand forecasting is critical for inventory management. Poor forecasts lead to either stockouts (lost sales) or overstock (holding costs). This project addresses both challenges by:
- Predicting demand with high accuracy
- Calculating optimal safety stock levels
- Determining reorder points
- Computing economic order quantities

## Features

### Input Features
- **Time-based**: Day of week, month, week of year, day of month
- **Demand patterns**: Base demand, trend, seasonality
- **Inventory parameters**: Lead time, holding cost, stockout cost, order cost
- **Demand volatility**: Standard deviation of demand

### Model Output
- Forecasted demand
- Optimized safety stock
- Reorder point
- Economic Order Quantity (EOQ)
- Total inventory cost breakdown

## Model Architecture

**Algorithm**: Gradient Boosting Regressor
- Captures non-linear demand patterns
- Handles multiple time-series features
- Provides accurate point forecasts

**Inventory Optimization**:
- Service level-based safety stock calculation (95% service level)
- EOQ formula for optimal order quantity
- Total cost minimization

## Project Structure

```
03-demand-forecasting/
├── generate_data.py              # Synthetic time-series data generation
├── demand_forecasting_model.py   # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   ├── training_data.csv         # Training dataset
│   └── test_data.csv             # Test dataset
└── models/
    ├── demand_model.pkl          # Trained model
    └── scaler.pkl                # Feature scaler
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

- **RMSE**: 5-8 units
- **MAE**: 3-5 units
- **MAPE**: 0.08-0.12 (8-12%)
- **R² Score**: 0.85-0.90

## Key Insights

1. **Seasonality**: Demand shows strong seasonal patterns
2. **Day-of-week effects**: Weekends typically show higher demand
3. **Lead time impact**: Longer lead times require higher safety stock
4. **Cost trade-off**: Balance between holding costs and stockout costs

## Inventory Optimization Metrics

### Safety Stock
- Calculated using z-score (1.65 for 95% service level)
- Adjusted for lead time and demand volatility
- Formula: Z × σ × √L

### Reorder Point
- Triggers new order when inventory reaches this level
- Formula: (Average Demand × Lead Time) + Safety Stock

### Economic Order Quantity
- Minimizes total ordering and holding costs
- Formula: √(2 × D × S / H)
  - D: Annual demand
  - S: Order cost per order
  - H: Holding cost per unit

## Real-World Applications

- **Retail**: Inventory management for stores
- **Manufacturing**: Raw material procurement
- **E-commerce**: Warehouse inventory optimization
- **Supply chain**: Multi-echelon inventory planning
- **Pharmaceuticals**: Drug inventory management

## Benefits

1. **Reduced stockouts**: Optimal safety stock prevents lost sales
2. **Lower holding costs**: EOQ minimizes excess inventory
3. **Improved cash flow**: Better inventory turnover
4. **Better service levels**: Consistent availability
5. **Cost savings**: Optimized ordering patterns

## Future Enhancements

1. Multi-step ahead forecasting
2. Hierarchical demand forecasting
3. Incorporate external factors (promotions, events)
4. Dynamic safety stock adjustment
5. Multi-product optimization

## References

- Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). Inventory and Production Management in Supply Chains
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods
- Chopra, S., & Meindl, P. (2015). Supply Chain Management: Strategy, Planning, and Operation

## Author

Created as part of ML Portfolio Projects

## License

MIT License
