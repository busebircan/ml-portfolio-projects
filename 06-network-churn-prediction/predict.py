"""
Prediction script for Network Churn Prediction
"""

import pandas as pd
from churn_prediction_model import NetworkChurnPredictor

def main():
    print("=" * 70)
    print("NETWORK CHURN PREDICTION - PREDICTION")
    print("=" * 70)
    
    # Load predictor and model
    print("\n1. Loading trained model...")
    predictor = NetworkChurnPredictor()
    predictor.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = predictor.generate_churn_data(n_samples=50, churn_rate=0.2, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    churn_probs, churn_classes = predictor.predict_churn(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Actual_Churn': test_df['churn'].values,
        'Churn_Probability': churn_probs,
        'Predicted_Churn': churn_classes,
        'Risk_Level': [predictor.get_churn_risk_level(p) for p in churn_probs],
        'Tenure_Months': test_df['tenure_months'].values,
        'Monthly_Charges': test_df['monthly_charges'].values,
        'Network_Satisfaction': test_df['network_satisfaction'].values,
        'Complaint_Count': test_df['complaint_count'].values,
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("CHURN PREDICTION RESULTS")
    print("=" * 70)
    print("\n" + results.head(20).to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    risk_dist = results['Risk_Level'].value_counts()
    print(f"\nRisk Distribution:")
    for risk_level in ['Low', 'Medium', 'High']:
        count = risk_dist.get(risk_level, 0)
        print(f"  {risk_level}: {count} ({count/len(results)*100:.1f}%)")
    
    print(f"\nAverage Churn Probability: {results['Churn_Probability'].mean():.4f}")
    print(f"Predicted Churn Rate: {results['Predicted_Churn'].mean():.2%}")
    print(f"Actual Churn Rate: {results['Actual_Churn'].mean():.2%}")
    
    # Save results
    results.to_csv('churn_predictions.csv', index=False)
    print("\n✓ Results saved to churn_predictions.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
