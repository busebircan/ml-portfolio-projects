"""
Training script for Hospital Readmission Prediction
"""

import pandas as pd
from readmission_model import ReadmissionPredictor

def main():
    print("=" * 70)
    print("HOSPITAL READMISSION PREDICTION - TRAINING")
    print("=" * 70)
    
    # Initialize predictor
    print("\n1. Initializing predictor...")
    predictor = ReadmissionPredictor()
    
    # Generate data
    print("\n2. Generating training data...")
    df = predictor.generate_readmission_data(n_samples=1000, readmission_rate=0.25, random_state=42)
    print(f"   ✓ Generated {len(df)} samples")
    print(f"   ✓ Readmission rate: {df['readmitted'].mean():.2%}")
    
    # Train model
    print("\n3. Training model...")
    metrics = predictor.train(df, test_size=0.2, random_state=42)
    
    # Save model
    print("\n4. Saving model...")
    predictor.save_model()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
