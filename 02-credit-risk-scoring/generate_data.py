"""
Credit Risk Scoring with Alternative Data - Data Generation
Generates synthetic data for credit risk assessment using traditional and alternative data
"""

import numpy as np
import pandas as pd
import os

def generate_credit_risk_data(n_samples=1000, random_state=42):
    """
    Generate synthetic data for credit risk scoring
    
    Features include:
    - Traditional: Income, employment history, credit history
    - Alternative: Social behavior, payment patterns, digital footprint
    """
    np.random.seed(random_state)
    
    data = {
        # Traditional Credit Features
        'age': np.random.uniform(18, 75, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'employment_years': np.random.exponential(8, n_samples),
        'existing_credit_accounts': np.random.poisson(3, n_samples),
        'credit_history_years': np.random.exponential(10, n_samples),
        'credit_utilization_ratio': np.random.uniform(0, 1, n_samples),
        'missed_payments_12m': np.random.poisson(0.5, n_samples),
        'missed_payments_24m': np.random.poisson(1, n_samples),
        'total_debt': np.random.exponential(30000, n_samples),
        'debt_to_income_ratio': np.random.uniform(0, 1, n_samples),
        
        # Alternative Data - Social Behavior
        'social_media_presence': np.random.binomial(1, 0.7, n_samples),
        'social_media_followers': np.random.exponential(500, n_samples),
        'social_media_posts_per_month': np.random.exponential(20, n_samples),
        'social_network_size': np.random.exponential(200, n_samples),
        'network_credit_quality': np.random.uniform(0, 1, n_samples),
        
        # Alternative Data - Digital Behavior
        'email_verification': np.random.binomial(1, 0.85, n_samples),
        'phone_verification': np.random.binomial(1, 0.80, n_samples),
        'identity_verification': np.random.binomial(1, 0.75, n_samples),
        'account_age_days': np.random.exponential(500, n_samples),
        'login_frequency_per_week': np.random.exponential(3, n_samples),
        'device_consistency': np.random.uniform(0, 1, n_samples),
        
        # Alternative Data - Transaction Behavior
        'transaction_frequency': np.random.exponential(50, n_samples),
        'transaction_variance': np.random.exponential(1000, n_samples),
        'average_transaction_amount': np.random.exponential(500, n_samples),
        'bill_payment_consistency': np.random.uniform(0, 1, n_samples),
        'on_time_payment_ratio': np.random.uniform(0, 1, n_samples),
        
        # Alternative Data - Online Reputation
        'online_reviews_count': np.random.poisson(5, n_samples),
        'average_review_rating': np.random.uniform(1, 5, n_samples),
        'dispute_history': np.random.binomial(1, 0.1, n_samples),
        'complaint_count': np.random.poisson(0.3, n_samples),
        
        # Employment and Stability
        'employment_type': np.random.choice(['employed', 'self-employed', 'unemployed'], n_samples, p=[0.7, 0.2, 0.1]),
        'job_stability_score': np.random.uniform(0, 1, n_samples),
        'income_stability': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate risk score based on features
    risk_score = (
        # Traditional factors (higher weight)
        (1 - df['on_time_payment_ratio']) * 0.25 +
        df['debt_to_income_ratio'] * 0.20 +
        (df['missed_payments_12m'] / 10) * 0.15 +
        (1 - df['credit_utilization_ratio']) * 0.05 +
        
        # Alternative factors (lower weight but important)
        (1 - df['email_verification']) * 0.05 +
        (1 - df['phone_verification']) * 0.05 +
        (1 - df['bill_payment_consistency']) * 0.10 +
        (1 - df['device_consistency']) * 0.05 +
        df['dispute_history'] * 0.05 +
        (df['complaint_count'] / 5) * 0.05
    )
    
    # Normalize risk score to 0-1
    risk_score = np.clip(risk_score, 0, 1)
    
    # Generate binary default indicator (1 = default, 0 = no default)
    # Higher risk score = higher probability of default
    default_probability = risk_score
    df['default'] = (np.random.random(n_samples) < default_probability).astype(int)
    
    df['risk_score'] = risk_score
    
    return df

def main():
    """Generate and save the dataset"""
    print("Generating Credit Risk Scoring dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    train_df = generate_credit_risk_data(n_samples=800, random_state=42)
    train_df.to_csv('data/training_data.csv', index=False)
    print(f"✓ Training data saved: {len(train_df)} samples")
    
    # Generate test data
    test_df = generate_credit_risk_data(n_samples=200, random_state=43)
    test_df.to_csv('data/test_data.csv', index=False)
    print(f"✓ Test data saved: {len(test_df)} samples")
    
    print("\nDataset Summary:")
    print(f"Features: {len(train_df.columns) - 2}")
    print(f"Default Rate: {train_df['default'].mean():.2%}")
    print(f"\nFeature Categories:")
    print(f"  - Traditional Credit: 10 features")
    print(f"  - Social Behavior: 5 features")
    print(f"  - Digital Behavior: 5 features")
    print(f"  - Transaction Behavior: 5 features")
    print(f"  - Online Reputation: 4 features")
    print(f"  - Employment & Stability: 3 features")

if __name__ == '__main__':
    main()
