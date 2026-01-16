"""
Prediction script for Call Drop Prediction
"""

import pandas as pd
from call_drop_model import CallDropPredictor

def main():
    print("=" * 70)
    print("CALL DROP PREDICTION - PREDICTION")
    print("=" * 70)
    
    # Load predictor and model
    print("\n1. Loading trained model...")
    predictor = CallDropPredictor()
    predictor.load_model()
    print("   ✓ Model loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df = predictor.generate_call_data(n_samples=50, drop_rate=0.15, random_state=99)
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    drop_probs, drop_classes = predictor.predict_drop_risk(test_df)
    
    # Prepare results
    results = pd.DataFrame({
        'Actual_Drop': test_df['call_dropped'].values,
        'Drop_Probability': drop_probs,
        'Predicted_Drop': drop_classes,
        'Risk_Level': [predictor.get_risk_level(p) for p in drop_probs],
        'Signal_Strength_dBm': test_df['signal_strength_dbm'].values,
        'SNR_dB': test_df['signal_to_noise_ratio'].values,
        'Packet_Loss_Pct': test_df['packet_loss_pct'].values,
        'Tower_Load_Pct': test_df['tower_load_pct'].values,
        'Latency_ms': test_df['latency_ms'].values,
    })
    
    # Display results
    print("\n" + "=" * 70)
    print("CALL DROP RISK PREDICTIONS")
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
    
    print(f"\nAverage Drop Probability: {results['Drop_Probability'].mean():.4f}")
    print(f"Predicted Drop Rate: {results['Predicted_Drop'].mean():.2%}")
    print(f"Actual Drop Rate: {results['Actual_Drop'].mean():.2%}")
    
    # Save results
    results.to_csv('drop_predictions.csv', index=False)
    print("\n✓ Results saved to drop_predictions.csv")
    
    print("\n" + "=" * 70)
    print("✓ Prediction completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
