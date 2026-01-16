"""
Training script for Multi-Store SKU Forecasting
"""

import pandas as pd
from multi_store_forecasting_model import MultiStoreSKUForecaster

def main():
    print("=" * 70)
    print("MULTI-STORE SKU FORECASTING - TRAINING")
    print("=" * 70)
    
    # Initialize forecaster
    print("\n1. Initializing forecaster...")
    forecaster = MultiStoreSKUForecaster()
    
    # Generate data
    print("\n2. Generating training data...")
    df = forecaster.generate_multi_store_data(n_stores=10, n_skus=5, n_days=365, random_state=42)
    print(f"   ✓ Generated {len(df)} records")
    print(f"   ✓ Stores: {df['store_id'].nunique()}, SKUs: {df['sku_id'].nunique()}")
    
    # Train models
    print("\n3. Training models...")
    metrics = forecaster.train(df, test_size=0.2, random_state=42)
    
    # Save models
    print("\n4. Saving models...")
    forecaster.save_models()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    
    summary_df = pd.DataFrame([\n        {\n            'Store_ID': store_id,\n            'R2_Score': f\"{metrics[store_id]['r2']:.4f}\",\n            'RMSE': f\"{metrics[store_id]['rmse']:.2f}\",\n            'MAPE': f\"{metrics[store_id]['mape']:.4f}\"\n        }\n        for store_id in sorted(metrics.keys())\n    ])\n    \n    print(\"\\n\" + summary_df.to_string(index=False))\n    \n    # Overall statistics\n    r2_scores = [metrics[s]['r2'] for s in metrics]\n    print(f\"\\nOverall R² Score: {sum(r2_scores)/len(r2_scores):.4f}\")\n    \n    print(\"\\n\" + \"=\" * 70)\n    print(\"✓ Training completed successfully!\")\n    print(\"=\" * 70)\n\nif __name__ == '__main__':\n    main()\n
