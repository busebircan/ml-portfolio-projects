"""
Credit Risk Scoring Model
Uses machine learning with alternative data to assess credit risk
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_score, recall_score, f1_score, classification_report
)
import joblib
import os

class CreditRiskScorer:
    """
    Assesses credit risk using traditional and alternative data
    Predicts probability of default
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
    def preprocess_data(self, df, fit=True):
        """Preprocess data for modeling"""
        df = df.copy()
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Store feature names
        if fit:
            self.feature_names = df.columns.tolist()
            # Remove target variables
            for col in ['default', 'risk_score']:
                if col in self.feature_names:
                    self.feature_names.remove(col)
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the credit risk model"""
        print("Training Credit Risk Scoring Model...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, fit=True)
        
        # Prepare features and target
        X = df_processed.drop(['default', 'risk_score'], axis=1)
        y = df_processed['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("  - Training Gradient Boosting Classifier...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        
        print("\n  - Top 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return metrics
    
    def predict_risk(self, features_df):
        """Predict risk probability for given features"""
        features_processed = self.preprocess_data(features_df, fit=False)
        X = features_processed.drop(['default', 'risk_score'], axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        risk_probability = self.model.predict_proba(X_scaled)[:, 1]
        risk_class = self.model.predict(X_scaled)
        
        return risk_probability, risk_class
    
    def get_risk_category(self, risk_probability):
        """Categorize risk into levels"""
        if risk_probability < 0.3:
            return 'Low'
        elif risk_probability < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def generate_risk_report(self, features_df):
        """Generate detailed risk assessment report"""
        features_processed = self.preprocess_data(features_df, fit=False)
        X = features_processed.drop(['default', 'risk_score'], axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        risk_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        risk_classes = self.model.predict(X_scaled)
        
        report = pd.DataFrame({
            'Risk_Probability': risk_probabilities,
            'Risk_Class': risk_classes,
            'Risk_Category': [self.get_risk_category(p) for p in risk_probabilities],
            'Age': features_df['age'].values,
            'Income': features_df['income'].values,
            'Debt_to_Income': features_df['debt_to_income_ratio'].values,
            'On_Time_Payment_Ratio': features_df['on_time_payment_ratio'].values,
            'Credit_History_Years': features_df['credit_history_years'].values,
        })
        
        return report
    
    def save_model(self, model_path='models'):
        """Save trained model"""
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.model, f'{model_path}/credit_risk_model.pkl')
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_path}/label_encoders.pkl')
        self.feature_importance.to_csv(f'{model_path}/feature_importance.csv', index=False)
        print(f"✓ Model saved to {model_path}/")
    
    def load_model(self, model_path='models'):
        """Load trained model"""
        self.model = joblib.load(f'{model_path}/credit_risk_model.pkl')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{model_path}/label_encoders.pkl')
        self.feature_importance = pd.read_csv(f'{model_path}/feature_importance.csv')
        print(f"✓ Model loaded from {model_path}/")

def main():
    """Main execution"""
    from generate_data import generate_credit_risk_data
    
    print("=" * 60)
    print("CREDIT RISK SCORING WITH ALTERNATIVE DATA")
    print("=" * 60)
    
    # Generate data
    df = generate_credit_risk_data(n_samples=1000)
    
    # Initialize and train model
    scorer = CreditRiskScorer()
    metrics = scorer.train(df)
    
    # Save model
    scorer.save_model()
    
    # Test predictions
    print("\n" + "=" * 60)
    print("RISK ASSESSMENT RESULTS")
    print("=" * 60)
    
    test_df = df.head(10).copy()
    report = scorer.generate_risk_report(test_df)
    
    print("\nSample Risk Assessment Report:")
    print(report.to_string(index=False))
    
    print("\n✓ Credit Risk Scoring Model training completed successfully!")

if __name__ == '__main__':
    main()
