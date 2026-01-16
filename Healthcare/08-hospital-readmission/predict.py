"""
Prediction script for Hospital Readmission Prediction
"""

import pandas as pd
from readmission_model import ReadmissionPredictor

def main():
    print("=" * 70)
    print("HOSPITAL READMISSION PREDICTION - PREDICTION")
    print("=" * 70)
    
    # Load predictor and model
    print("\n1. Loading trained model...")
    predictor = ReadmissionPredictor()
    predictor.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = predictor.generate_readmission_data(n_samples=30, readmission_rate=0.25, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    readmission_probs, readmission_classes = predictor.predict_readmission_risk(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Actual_Readmitted': test_df['readmitted'].values,
        'Readmission_Probability': readmission_probs,
        'Predicted_Readmitted': readmission_classes,
        'Age': test_df['age'].values,
        'Charlson_Index': test_df['charlson_index'].values,
        'HbA1c_Level': test_df['hba1c_level'].values,
        'Num_Medications': test_df['num_medications'].values,
        'Prior_Readmissions_1yr': test_df['prior_readmissions_1yr'].values,
        'Has_Caregiver': test_df['has_caregiver'].values,
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("READMISSION RISK PREDICTIONS")
    print("=" * 70)
    print("\n" + results.head(20).to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nAverage Readmission Probability: {results['Readmission_Probability'].mean():.4f}")
    print(f"Predicted Readmission Rate: {results['Predicted_Readmitted'].mean():.2%}")
    print(f"Actual Readmission Rate: {results['Actual_Readmitted'].mean():.2%}")
    
    # Risk stratification
    print(f"\nRisk Stratification:")
    high_risk = (results['Readmission_Probability'] > 0.6).sum()
    medium_risk = ((results['Readmission_Probability'] >= 0.3) & (results['Readmission_Probability'] <= 0.6)).sum()
    low_risk = (results['Readmission_Probability'] < 0.3).sum()
    
    print(f"  High Risk (>0.6): {high_risk} ({high_risk/len(results)*100:.1f}%)")
    print(f"  Medium Risk (0.3-0.6): {medium_risk} ({medium_risk/len(results)*100:.1f}%)")
    print(f"  Low Risk (<0.3): {low_risk} ({low_risk/len(results)*100:.1f}%)")
    
    # Save results
    results.to_csv('readmission_predictions.csv', index=False)
    print("\n✓ Results saved to readmission_predictions.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
