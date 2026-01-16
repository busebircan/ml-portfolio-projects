"""
Training script for Price Elasticity Analysis
"""

import pandas as pd
from price_elasticity_model import PriceElasticityAnalyzer

def main():
    print("=" * 70)
    print("PRICE ELASTICITY ANALYSIS - TRAINING")
    print("=" * 70)
    
    # Initialize analyzer
    print("\n1. Initializing analyzer...")
    analyzer = PriceElasticityAnalyzer()
    
    # Generate data
    print("\n2. Generating training data...")
    df = analyzer.generate_elasticity_data(n_samples=500, random_state=42)
    print(f"   ✓ Generated {len(df)} training samples")
    print(f"   ✓ Categories: {df['category'].nunique()}")
    
    # Train models
    print("\n3. Training elasticity models...")
    metrics = analyzer.train(df, test_size=0.2, random_state=42)
    
    # Save models
    print("\n4. Saving models...")
    analyzer.save_models()
    
    # Display results
    print("\n" + "=" * 70)
    print("ELASTICITY SUMMARY")
    print("=" * 70)
    
    summary_df = pd.DataFrame([\n        {\n            'Category': cat,\n            'Price_Elasticity': f\"{metrics[cat]['elasticity']:.3f}\",\n            'Elasticity_Type': 'Elastic' if metrics[cat]['elasticity'] < -1 else 'Inelastic',\n            'R2_Score': f\"{metrics[cat]['r2']:.4f}\",\n            'RMSE': f\"{metrics[cat]['rmse']:.2f}\"\n        }\n        for cat in metrics\n    ])\n    \n    print(\"\\n\" + summary_df.to_string(index=False))\n    \n    print("\n\" + \"=\" * 70)\n    print(\"✓ Training completed successfully!\")\n    print(\"=\" * 70)\n\nif __name__ == '__main__':\n    main()\n
