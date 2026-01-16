"""
Prediction script for Credit Risk Scoring Model
"""

import pandas as pd
from credit_risk_model import CreditRiskScorer
from generate_data import generate_credit_risk_data

def main():
    print("=" * 70)
    print("CREDIT RISK SCORING - PREDICTION")
    print("=" * 70)
    
    # Load scorer and model
    print("\n1. Loading trained model...")
    scorer = CreditRiskScorer()
    scorer.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = generate_credit_risk_data(n_samples=20, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    report = scorer.generate_risk_report(test_df)
    
    # Display results
    print("\n" + "=" * 70)
    print("CREDIT RISK ASSESSMENT REPORT")
    print("=" * 70)
    print(report.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    risk_dist = report['Risk_Category'].value_counts()
    print(f"\nRisk Distribution:")
    for category, count in risk_dist.items():
        print(f"  {category}: {count} ({count/len(report)*100:.1f}%)")
    
    print(f"\nAverage Risk Probability: {report['Risk_Probability'].mean():.4f}")
    print(f"Min Risk Probability: {report['Risk_Probability'].min():.4f}")
    print(f"Max Risk Probability: {report['Risk_Probability'].max():.4f}")
    
    print(f"\nDefault Rate in Sample: {report['Risk_Class'].mean():.2%}")
    
    # Save results
    report.to_csv('assessment_results.csv', index=False)
    print("\n✓ Results saved to assessment_results.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
