"""
Prediction script for Early Sepsis Detection
"""

import pandas as pd
from sepsis_model import SepsisDetector

def main():
    print("=" * 70)
    print("EARLY SEPSIS DETECTION - PREDICTION")
    print("=" * 70)
    
    # Load detector and model
    print("\n1. Loading trained model...")
    detector = SepsisDetector()
    detector.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = detector.generate_sepsis_data(n_samples=30, sepsis_rate=0.15, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    sepsis_probs, sepsis_classes = detector.predict_sepsis_risk(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Actual_Sepsis': test_df['sepsis'].values,
        'Sepsis_Probability': sepsis_probs,
        'Predicted_Sepsis': sepsis_classes,
        'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in sepsis_probs],
        'Temperature_Change': test_df['temp_change'].values,
        'HR_Change': test_df['hr_change'].values,
        'Lactate': test_df['lactate'].values,
        'Procalcitonin': test_df['procalcitonin'].values,
        'WBC_Count': test_df['wbc_count'].values,
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("SEPSIS RISK PREDICTIONS")
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
    
    print(f"\nAverage Sepsis Probability: {results['Sepsis_Probability'].mean():.4f}")
    print(f"Predicted Sepsis Rate: {results['Predicted_Sepsis'].mean():.2%}")
    print(f"Actual Sepsis Rate: {results['Actual_Sepsis'].mean():.2%}")
    
    # Save results
    results.to_csv('sepsis_predictions.csv', index=False)
    print("\n✓ Results saved to sepsis_predictions.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
