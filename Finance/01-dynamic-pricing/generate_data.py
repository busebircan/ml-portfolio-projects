"""
Dynamic Pricing Optimization - Data Generation
Generates synthetic data for dynamic pricing based on demand, competition, and inventory
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_dynamic_pricing_data(n_samples=1000, random_state=42):
    """
    Generate synthetic data for dynamic pricing optimization
    
    Features:
    - Demand patterns (seasonal, trend)
    - Competition pricing
    - Inventory levels
    - Customer segments
    - Time-based factors
    """
    np.random.seed(random_state)
    
    # Time features
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=int(i)) for i in np.random.uniform(0, 365, n_samples)]
    
    data = {
        'date': dates,
        'day_of_week': [d.dayofweek for d in dates],
        'month': [d.month for d in dates],
        'is_weekend': [1 if d.dayofweek >= 5 else 0 for d in dates],
        'is_holiday': np.random.binomial(1, 0.05, n_samples),
        
        # Demand features
        'base_demand': np.random.gamma(shape=2, scale=50, size=n_samples),
        'demand_shock': np.random.normal(0, 10, n_samples),
        'seasonal_factor': np.random.uniform(0.8, 1.3, n_samples),
        
        # Competition features
        'competitor_price': np.random.uniform(20, 100, n_samples),
        'competitor_count': np.random.poisson(3, n_samples),
        'market_share': np.random.uniform(0.1, 0.5, n_samples),
        
        # Inventory features
        'inventory_level': np.random.gamma(shape=3, scale=100, size=n_samples),
        'inventory_cost': np.random.uniform(5, 30, n_samples),
        'stockout_penalty': np.random.uniform(50, 200, n_samples),
        
        # Customer features
        'customer_segment': np.random.choice(['premium', 'standard', 'budget'], n_samples),
        'price_elasticity': np.random.uniform(-2, -0.5, n_samples),
        'customer_lifetime_value': np.random.exponential(500, n_samples),
        
        # Cost features
        'production_cost': np.random.uniform(10, 40, n_samples),
        'distribution_cost': np.random.uniform(2, 15, n_samples),
        'marketing_spend': np.random.uniform(0, 50, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate optimal price based on features
    df['optimal_price'] = (
        df['production_cost'] + df['distribution_cost'] +
        np.where(df['customer_segment'] == 'premium', 30, 
                np.where(df['customer_segment'] == 'standard', 15, 5)) +
        (df['competitor_price'] - 50) * 0.3 +
        np.where(df['inventory_level'] > 500, -5, 
                np.where(df['inventory_level'] < 100, 10, 0)) +
        np.random.normal(0, 3, n_samples)
    )
    
    # Generate demand based on price
    df['demand'] = (
        df['base_demand'] * df['seasonal_factor'] +
        df['demand_shock'] +
        (df['optimal_price'] - 50) * df['price_elasticity'] +
        np.random.normal(0, 5, n_samples)
    )
    df['demand'] = df['demand'].clip(lower=0)
    
    # Generate revenue
    df['revenue'] = df['optimal_price'] * df['demand']
    
    # Generate profit
    df['profit'] = (
        df['revenue'] - 
        (df['production_cost'] + df['distribution_cost']) * df['demand'] -
        df['marketing_spend']
    )
    
    return df

def main():
    """Generate and save the dataset"""
    print("Generating Dynamic Pricing dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    train_df = generate_dynamic_pricing_data(n_samples=800, random_state=42)
    train_df.to_csv('data/training_data.csv', index=False)
    print(f"✓ Training data saved: {len(train_df)} samples")
    
    # Generate test data
    test_df = generate_dynamic_pricing_data(n_samples=200, random_state=43)
    test_df.to_csv('data/test_data.csv', index=False)
    print(f"✓ Test data saved: {len(test_df)} samples")
    
    print("\nDataset Summary:")
    print(f"Features: {list(train_df.columns)}")
    print(f"\nTraining Data Statistics:")
    print(train_df[['optimal_price', 'demand', 'revenue', 'profit']].describe())

if __name__ == '__main__':
    main()
