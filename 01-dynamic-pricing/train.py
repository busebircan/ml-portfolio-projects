"""
Training script for Dynamic Pricing Model
"""

import pandas as pd
from dynamic_pricing_model import DynamicPricingOptimizer
from generate_data import generate_dynamic_pricing_data

def main():
    print("=" * 70)
    print("DYNAMIC PRICING OPTIMIZATION - TRAINING")
    print("=" * 70)
    
    # Generate training data
    print("\n1. Generating training data...")
    df = generate_dynamic_pricing_data(n_samples=1000, random_state=42)
    print(f"   ✓ Generated {len(df)} training samples")
    
    # Initialize optimizer
    print("\n2. Initializing optimizer...")
    optimizer = DynamicPricingOptimizer()
    
    # Train model
    print("\n3. Training models...")
    metrics = optimizer.train(df, test_size=0.2, random_state=42)
    
    # Save models
    print("\n4. Saving models...")
    optimizer.save_model()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print("\nPrice Prediction Model:")
    for metric, value in metrics['price_model'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nDemand Prediction Model:")
    for metric, value in metrics['demand_model'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
