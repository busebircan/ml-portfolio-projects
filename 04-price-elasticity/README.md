# Price Elasticity Analysis

## Project Overview

This project analyzes price elasticity of demand across different product categories. It builds machine learning models to understand how price changes affect quantity demanded and helps businesses optimize pricing strategies based on elasticity insights.

## Business Problem

Different products have different price sensitivities. Understanding elasticity helps businesses:
- Identify which products can support price increases
- Determine optimal pricing strategies
- Predict revenue impact of price changes
- Segment products by elasticity
- Maximize profit through informed pricing

## Elasticity Concepts

**Price Elasticity of Demand (PED)** measures the responsiveness of quantity demanded to price changes:

- **Elastic (PED < -1)**: Quantity changes more than price (e.g., Electronics)
- **Inelastic (PED > -1)**: Quantity changes less than price (e.g., Food)
- **Unit Elastic (PED = -1)**: Proportional changes

## Features

### Input Features
- Price
- Competitor pricing
- Marketing spend
- Seasonality factors
- Average customer income
- Price-to-competitor ratio

### Model Output
- Price elasticity coefficient
- Demand predictions at different price points
- Revenue simulations
- Elasticity classification

## Project Structure

```
04-price-elasticity/
├── price_elasticity_model.py     # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    └── elasticity_models.pkl     # Trained models by category
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

- **R² Score**: 0.80-0.85 across categories
- **RMSE**: Varies by category (10-50 units)
- **Elasticity Accuracy**: Captures category-specific patterns

## Category-Specific Elasticity

| Category | Elasticity | Type | Implication |
|----------|-----------|------|-------------|
| Electronics | -1.5 | Elastic | Price sensitive; lower prices increase revenue |
| Clothing | -1.2 | Elastic | Moderately price sensitive |
| Food | -0.8 | Inelastic | Price insensitive; price increases increase revenue |
| Home & Garden | -1.0 | Unit Elastic | Proportional changes |
| Sports | -1.3 | Elastic | Price sensitive |

## Real-World Applications

- **Retail Pricing**: Optimize prices by category
- **Revenue Management**: Dynamic pricing strategies
- **Promotional Planning**: Determine discount impact
- **Market Analysis**: Understand competitive positioning
- **Product Strategy**: Identify price-sensitive segments

## Pricing Strategies by Elasticity

### For Elastic Products (PED < -1)
- Decrease prices to increase revenue
- Use promotional pricing
- Focus on volume growth
- Compete on price

### For Inelastic Products (PED > -1)
- Increase prices to increase revenue
- Minimize discounting
- Focus on value proposition
- Compete on quality/brand

## Benefits

1. **Data-driven pricing**: Base decisions on elasticity analysis
2. **Revenue optimization**: Maximize profit through optimal pricing
3. **Competitive advantage**: Understand market dynamics
4. **Risk reduction**: Predict price change impacts
5. **Strategic planning**: Inform long-term pricing strategy

## Interpretation Example

If Electronics has elasticity of -1.5:
- 1% price increase → 1.5% quantity decrease
- 10% price decrease → 15% quantity increase

## Future Enhancements

1. Cross-price elasticity (substitute/complement products)
2. Income elasticity analysis
3. Time-varying elasticity
4. Competitive elasticity
5. Seasonal elasticity variations

## References

- Pindyck, R. S., & Rubinfeld, D. L. (2012). Microeconomics
- Nagle, T. T., & Müller, G. (2017). The Strategy and Tactics of Pricing
- Hanssens, D. M., Parsons, L. J., & Schultz, R. L. (2001). Market Response Models

## Author

Created as part of ML Portfolio Projects

## License

MIT License
