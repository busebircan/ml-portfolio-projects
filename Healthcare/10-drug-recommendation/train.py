"""
Training script for Drug Recommendation System
"""

import pandas as pd
from drug_recommendation_model import DrugRecommender

def main():
    print("=" * 70)
    print("DRUG RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 70)
    
    # Initialize recommender
    print("\n1. Initializing recommender...")
    recommender = DrugRecommender()
    
    # Generate data
    print("\n2. Generating training data...")
    df = recommender.generate_drug_recommendation_data(n_samples=1000, random_state=42)
    print(f"   ✓ Generated {len(df)} samples")
    print(f"   ✓ Drugs available: {len(recommender.drug_list)}")
    
    # Train models
    print("\n3. Training models...")
    metrics = recommender.train(df, test_size=0.2, random_state=42)
    
    # Save models
    print("\n4. Saving models...")
    recommender.save_models()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    
    print("\nPer-Drug Accuracy:")
    for drug, metric in metrics.items():
        print(f"  {drug}: {metric['accuracy']:.4f}")
    
    avg_accuracy = sum([m['accuracy'] for m in metrics.values()]) / len(metrics)
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
