"""
Demand Forecasting with Inventory Optimization - Data Generation
Generates synthetic time-series data for demand forecasting and inventory optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_demand_forecasting_data(n_days=365, n_products=5, random_state=42):
    """
    Generate synthetic time-series demand data with inventory optimization
    """
    np.random.seed(random_state)
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for product_id in range(1, n_products + 1):
        # Product-specific parameters
        base_demand = np.random.uniform(50, 200)
        trend = np.random.uniform(-0.5, 0.5)
        seasonality_amplitude = np.random.uniform(20, 80)
        
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            
            # Time features
            day_of_week = date.dayofweek
            month = date.month
            week_of_year = date.isocalendar()[1]
            
            # Demand components
            trend_component = trend * day
            seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * day / 365)
            day_of_week_effect = np.where(day_of_week >= 5, 20, -10)  # Weekend boost
            random_shock = np.random.normal(0, 15)
            
            # Total demand
            demand = base_demand + trend_component + seasonal_component + day_of_week_effect + random_shock
            demand = max(0, demand)
            
            # Inventory dynamics
            lead_time = np.random.randint(3, 14)
            holding_cost = np.random.uniform(1, 5)
            stockout_cost = np.random.uniform(20, 100)
            
            # Reorder point and safety stock calculation
            safety_stock = 1.65 * np.sqrt(lead_time) * np.std([demand, demand * 0.8, demand * 1.2])
            reorder_point = demand * lead_time + safety_stock
            
            # Economic order quantity (EOQ)
            order_cost = np.random.uniform(10, 50)
            annual_demand = demand * 365
            eoq = np.sqrt(2 * annual_demand * order_cost / holding_cost)
            
            data.append({
                'date': date,
                'product_id': product_id,
                'day_of_week': day_of_week,
                'month': month,
                'week_of_year': week_of_year,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'day_of_month': date.day,
                
                # Demand
                'demand': demand,
                'base_demand': base_demand,
                'trend_component': trend_component,
                'seasonal_component': seasonal_component,
                
                # Inventory parameters
                'lead_time_days': lead_time,
                'holding_cost_per_unit': holding_cost,
                'stockout_cost_per_unit': stockout_cost,
                'order_cost': order_cost,
                
                # Calculated metrics
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'eoq': eoq,
                'demand_std': np.std([demand, demand * 0.8, demand * 1.2]),
            })
    
    df = pd.DataFrame(data)
    return df

def main():
    """Generate and save the dataset"""
    print("Generating Demand Forecasting dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    train_df = generate_demand_forecasting_data(n_days=365, n_products=5, random_state=42)
    train_df.to_csv('data/training_data.csv', index=False)
    print(f"✓ Training data saved: {len(train_df)} samples")
    
    # Generate test data
    test_df = generate_demand_forecasting_data(n_days=90, n_products=5, random_state=43)
    test_df.to_csv('data/test_data.csv', index=False)
    print(f"✓ Test data saved: {len(test_df)} samples")
    
    print("\nDataset Summary:")
    print(f"Time Period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Products: {train_df['product_id'].nunique()}")
    print(f"Average Daily Demand: {train_df['demand'].mean():.2f} units")
    print(f"Demand Std Dev: {train_df['demand'].std():.2f} units")

if __name__ == '__main__':
    main()
