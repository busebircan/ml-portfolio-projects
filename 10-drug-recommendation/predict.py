"""
Prediction script for Drug Recommendation System
"""

import pandas as pd
from drug_recommendation_model import DrugRecommender

def main():
    print("=" * 70)
    print("DRUG RECOMMENDATION SYSTEM - PREDICTION")
    print("=" * 70)
    
    # Load recommender and models
    print("\n1. Loading trained models...")
    recommender = DrugRecommender()
    recommender.load_models()
    print("   ✓ Models loaded successfully")
    
    # Generate test data
    print("\n2. Generating test data...")
    test_df_full = recommender.generate_drug_recommendation_data(n_samples=20, random_state=99)
    test_df = test_df_full[recommender.feature_names]
    print(f"   ✓ Generated {len(test_df)} test samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    recommendations, confidences = recommender.recommend_drugs(test_df, confidence_threshold=0.5)
    
    # Prepare results
    results = []
    for i, (rec, conf) in enumerate(zip(recommendations, confidences)):
        patient_data = test_df_full.iloc[i]\n        results.append({\n            'Patient_ID': i + 1,\n            'Age': int(patient_data['age']),\n            'Conditions': ', '.join([\n                col.replace('_', ' ').title() \n                for col in ['diabetes', 'hypertension', 'heart_disease', 'asthma', 'copd']\n                if patient_data[col] == 1\n            ]) or 'None',\n            'Recommended_Drugs': ', '.join(rec) if rec else 'None',\n            'Num_Recommendations': len(rec)\n        })\n    \n    results_df = pd.DataFrame(results)\n    \n    # Display results\n    print("\n" + "=" * 70)\n    print("DRUG RECOMMENDATIONS\")\n    print(\"=\" * 70)\n    print(\"\\n\" + results_df.to_string(index=False))\n    \n    # Summary statistics\n    print(\"\\n\" + \"=\" * 70)\n    print(\"SUMMARY STATISTICS\")\n    print(\"=\" * 70)\n    \n    print(f\"\\nAverage Recommendations per Patient: {results_df['Num_Recommendations'].mean():.2f}\")\n    print(f\"Patients with Recommendations: {(results_df['Num_Recommendations'] > 0).sum()} ({(results_df['Num_Recommendations'] > 0).sum()/len(results_df)*100:.1f}%)\")\n    print(f\"Patients with No Recommendations: {(results_df['Num_Recommendations'] == 0).sum()} ({(results_df['Num_Recommendations'] == 0).sum()/len(results_df)*100:.1f}%)\")\n    \n    # Drug frequency\n    print(f\"\\nMost Recommended Drugs:\")\n    all_drugs = []\n    for rec in recommendations:\n        all_drugs.extend(rec)\n    \n    from collections import Counter\n    drug_counts = Counter(all_drugs)\n    for drug, count in drug_counts.most_common(5):\n        print(f\"  {drug}: {count} times ({count/len(recommendations)*100:.1f}%)\")\n    \n    # Save results\n    results_df.to_csv('drug_recommendations.csv', index=False)\n    print(\"\\n✓ Results saved to drug_recommendations.csv\")\n    \n    print(\"\\n\" + \"=\" * 70)\n    print(\"✓ Prediction completed successfully!\")\n    print(\"=\" * 70)\n\nif __name__ == '__main__':\n    main()\n
