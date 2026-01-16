"""
Training script for Early Sepsis Detection
"""

import pandas as pd
from sepsis_model import SepsisDetector

def main():
    print("=" * 70)
    print("EARLY SEPSIS DETECTION - TRAINING")
    print("=" * 70)
    
    # Initialize detector
    print("\n1. Initializing detector...")
    detector = SepsisDetector()
    
    # Generate data
    print("\n2. Generating training data...")
    df = detector.generate_sepsis_data(n_samples=1000, sepsis_rate=0.15, random_state=42)
    print(f"   ✓ Generated {len(df)} samples")
    print(f"   ✓ Sepsis rate: {df['sepsis'].mean():.2%}")
    
    # Train model
    print("\n3. Training model...")
    metrics = detector.train(df, test_size=0.2, random_state=42)
    
    # Save model
    print("\n4. Saving model...")
    detector.save_model()
    
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
