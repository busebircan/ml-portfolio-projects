"""
Prediction script for Price Elasticity Analysis
"""

import pandas as pd
from price_elasticity_model import PriceElasticityAnalyzer

def main():
    print("=" * 70)
    print("PRICE ELASTICITY ANALYSIS - PREDICTION")
    print("=" * 70)
    
    # Load analyzer and models
    print("\n1. Loading trained models...")
    analyzer = PriceElasticityAnalyzer()
    analyzer.load_models()
    print("   ✓ Models loaded successfully")
    
    # Simulate price changes for each category
    print("\n2. Simulating price changes...")
    
    categories = ['Electronics', 'Clothing', 'Food', 'Home & Garden', 'Sports']
    base_prices = [500, 75, 10, 200, 150]
    
    all_results = []
    
    for category, base_price in zip(categories, base_prices):
        print(f"\n   {category}:")
        
        # Elasticity for this category
        elasticity = analyzer.get_elasticity(category)
        print(f"     Price Elasticity: {elasticity:.3f}\")
        
        # Simulate different price changes
        price_changes = [-0.20, -0.10, 0, 0.10, 0.20]  # -20% to +20%
        
        simulation = analyzer.simulate_price_change(\n            category=category,\n            base_price=base_price,\n            price_changes=price_changes,\n            competitor_price=base_price * 0.95,\n            marketing_spend=1000,\n            seasonality_factor=1.0,\n            avg_income=75000\n        )\n        \n        simulation['category'] = category\n        simulation['base_price'] = base_price\n        all_results.append(simulation)\n        \n        print(\"     Price Scenarios:\")\n        for idx, row in simulation.iterrows():\n            print(f\"       {row['price_change_%']:+.0f}% → Price: ${row['new_price']:.2f}, \"\n                  f\"Quantity: {row['quantity']:.0f}, Revenue: ${row['revenue']:.0f}\")\n    \n    # Combine results\n    results_df = pd.concat(all_results, ignore_index=True)\n    \n    # Display summary\n    print(\"\\n\" + \"=\" * 70)\n    print(\"PRICE ELASTICITY SIMULATION RESULTS\")\n    print(\"=\" * 70)\n    print(\"\\n\" + results_df.to_string(index=False))\n    \n    # Save results\n    results_df.to_csv('elasticity_simulation_results.csv', index=False)\n    print(\"\\n✓ Results saved to elasticity_simulation_results.csv\")\n    \n    print(\"\\n\" + \"=\" * 70)\n    print(\"✓ Prediction completed successfully!\")\n    print(\"=\" * 70)\n\nif __name__ == '__main__':\n    main()\n
