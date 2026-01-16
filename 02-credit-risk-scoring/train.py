"""
Training script for Credit Risk Scoring Model
"""

import pandas as pd
from credit_risk_model import CreditRiskScorer
from generate_data import generate_credit_risk_data

def main():
    print("=" * 70)
    print("CREDIT RISK SCORING - TRAINING")
    print("=" * 70)
    
    # Generate training data
    print("\n1. Generating training data...")
    df = generate_credit_risk_data(n_samples=1000, random_state=42)
    print(f"   ✓ Generated {len(df)} training samples")
    print(f"   ✓ Default rate: {df['default'].mean():.2%}")
    
    # Initialize scorer
    print("\n2. Initializing credit risk scorer...")
    scorer = CreditRiskScorer()
    
    # Train model
    print("\n3. Training model...")
    metrics = scorer.train(df, test_size=0.2, random_state=42)
    
    # Save model
    print("\n4. Saving model...")
    scorer.save_model()
    
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
