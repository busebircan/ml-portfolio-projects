"""
Prediction script for Multi-Store SKU Forecasting
"""

import pandas as pd
import numpy as np
from multi_store_forecasting_model import MultiStoreSKUForecaster
from datetime import datetime, timedelta

def main():
    print("=" * 70)
    print("MULTI-STORE SKU FORECASTING - PREDICTION")
    print("=" * 70)
    
    # Load forecaster and models
    print("\n1. Loading trained models...")
    forecaster = MultiStoreSKUForecaster()
    forecaster.load_models()
    print("   ✓ Models loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    df = forecaster.generate_multi_store_data(n_stores=10, n_skus=5, n_days=30, random_state=99)
    
    # Encode categorical variables
    df['store_size_encoded'] = pd.factorize(df['store_size'])[0]
    df['sku_category_encoded'] = pd.factorize(df['sku_category'])[0]
    print(f"   ✓ Generated {len(df)} test records")
    
    # Make predictions
    print("\n3. Making predictions...")
    
    all_predictions = []\n    \n    for store_id in df['store_id'].unique():\n        store_data = df[df['store_id'] == store_id].copy()\n        \n        # Make forecast\n        forecast = forecaster.forecast(store_id, store_data)\n        \n        # Add to results\n        store_data['forecast'] = forecast\n        store_data['forecast_error'] = abs(store_data['sales'] - forecast)\n        store_data['forecast_error_pct'] = (store_data['forecast_error'] / store_data['sales'] * 100).replace([np.inf, -np.inf], 0)\n        \n        all_predictions.append(store_data)\n    \n    results_df = pd.concat(all_predictions, ignore_index=True)\n    \n    # Select key columns\n    display_cols = ['date', 'store_id', 'sku_id', 'sales', 'forecast', 'forecast_error', 'forecast_error_pct']\n    results_display = results_df[display_cols].head(30)\n    \n    # Display results\n    print(\"\\n\" + \"=\" * 70)\n    print(\"FORECAST RESULTS\")\n    print(\"=\" * 70)\n    print(\"\\n\" + results_display.to_string(index=False))\n    \n    # Summary statistics\n    print(\"\\n\" + \"=\" * 70)\n    print(\"SUMMARY STATISTICS\")\n    print(\"=\" * 70)\n    \n    print(f\"\\nAverage Actual Sales: {results_df['sales'].mean():.2f} units\")\n    print(f\"Average Forecast: {results_df['forecast'].mean():.2f} units\")\n    print(f\"Average Forecast Error: {results_df['forecast_error'].mean():.2f} units\")\n    print(f\"Average MAPE: {results_df['forecast_error_pct'].mean():.2f}%\")\n    \n    print(f\"\\nBy Store:\")\n    store_summary = results_df.groupby('store_id').agg({\n        'sales': 'mean',\n        'forecast': 'mean',\n        'forecast_error': 'mean',\n        'forecast_error_pct': 'mean'\n    }).round(2)\n    print(store_summary)\n    \n    # Save results\n    results_df.to_csv('forecast_results.csv', index=False)\n    print(\"\\n✓ Results saved to forecast_results.csv\")\n    \n    print(\"\\n\" + \"=\" * 70)\n    print(\"✓ Prediction completed successfully!\")\n    print(\"=\" * 70)\n\nif __name__ == '__main__':\n    main()\n
