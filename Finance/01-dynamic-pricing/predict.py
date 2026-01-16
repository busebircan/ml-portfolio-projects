"""
Prediction script for Dynamic Pricing Model
"""

import pandas as pd
from dynamic_pricing_model import DynamicPricingOptimizer
from generate_data import generate_dynamic_pricing_data

def main():
    print("=" * 70)
    print("DYNAMIC PRICING OPTIMIZATION - PREDICTION")
    print("=" * 70)
    
    # Load optimizer and models
    print("\n1. Loading trained models...")
    optimizer = DynamicPricingOptimizer()
    optimizer.load_model()
    print("   ✓ Models loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = generate_dynamic_pricing_data(n_samples=20, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    optimized_prices, predicted_demands = optimizer.optimize_pricing_strategy(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Original_Price': test_df['optimal_price'].values,
        'Optimized_Price': optimized_prices,
        'Price_Change_%': ((optimized_prices - test_df['optimal_price'].values) / 
                          test_df['optimal_price'].values * 100),
        'Predicted_Demand': predicted_demands,
        'Inventory_Level': test_df['inventory_level'].values,
        'Competitor_Price': test_df['competitor_price'].values,
        'Customer_Segment': test_df['customer_segment'].values
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("PRICING OPTIMIZATION RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nAverage Price Change: {results['Price_Change_%'].mean():.2f}%")
    print(f"Max Price Increase: {results['Price_Change_%'].max():.2f}%")
    print(f"Max Price Decrease: {results['Price_Change_%'].min():.2f}%")
    print(f"\nAverage Original Price: ${results['Original_Price'].mean():.2f}")
    print(f"Average Optimized Price: ${results['Optimized_Price'].mean():.2f}")
    print(f"Average Predicted Demand: {results['Predicted_Demand'].mean():.2f} units")
    
    # Save results
    results.to_csv('prediction_results.csv', index=False)
    print("\n✓ Results saved to prediction_results.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
