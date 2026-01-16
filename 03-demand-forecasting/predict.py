"""
Prediction script for Demand Forecasting Model
"""

import pandas as pd
from demand_forecasting_model import DemandForecaster
from generate_data import generate_demand_forecasting_data

def main():
    print("=" * 70)
    print("DEMAND FORECASTING - PREDICTION & INVENTORY OPTIMIZATION")
    print("=" * 70)
    
    # Load forecaster and model
    print("\n1. Loading trained model...")
    forecaster = DemandForecaster()
    forecaster.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = generate_demand_forecasting_data(n_days=30, n_products=5, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions and optimize inventory
    print("\n3. Making predictions and optimizing inventory...")
    optimization = forecaster.optimize_inventory(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Product_ID': test_df['product_id'].values,
        'Date': test_df['date'].values,
        'Actual_Demand': test_df['demand'].values,
        'Forecasted_Demand': optimization['forecasted_demand'],
        'Safety_Stock': optimization['safety_stock'],
        'Reorder_Point': optimization['reorder_point'],
        'EOQ': optimization['eoq'],
        'Total_Inventory_Cost': optimization['total_inventory_cost']
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("DEMAND FORECAST & INVENTORY OPTIMIZATION RESULTS")
    print("=" * 70)
    print(results.head(15).to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nDemand Forecasting:")
    print(f"  Average Actual Demand: {results['Actual_Demand'].mean():.2f} units")
    print(f"  Average Forecasted Demand: {results['Forecasted_Demand'].mean():.2f} units")
    print(f"  Forecast Error: {abs(results['Actual_Demand'].mean() - results['Forecasted_Demand'].mean()):.2f} units")
    
    print(f"\nInventory Optimization:")
    print(f"  Average Safety Stock: {results['Safety_Stock'].mean():.2f} units")
    print(f"  Average Reorder Point: {results['Reorder_Point'].mean():.2f} units")
    print(f"  Average EOQ: {results['EOQ'].mean():.2f} units")
    print(f"  Total Inventory Cost: ${results['Total_Inventory_Cost'].sum():.2f}")
    
    # Save results
    results.to_csv('forecast_results.csv', index=False)
    print("\n✓ Results saved to forecast_results.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
