"""
Training script for Demand Forecasting Model
"""

import pandas as pd
from demand_forecasting_model import DemandForecaster
from generate_data import generate_demand_forecasting_data

def main():
    print("=" * 70)
    print("DEMAND FORECASTING - TRAINING")
    print("=" * 70)
    
    # Generate training data
    print("\n1. Generating training data...")
    df = generate_demand_forecasting_data(n_days=365, n_products=5, random_state=42)
    print(f"   ✓ Generated {len(df)} training samples")
    
    # Initialize forecaster
    print("\n2. Initializing demand forecaster...")
    forecaster = DemandForecaster()
    
    # Train model
    print("\n3. Training model...")
    metrics = forecaster.train(df, test_size=0.2, random_state=42)
    
    # Save model
    print("\n4. Saving model...")
    forecaster.save_model()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"\nRMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
