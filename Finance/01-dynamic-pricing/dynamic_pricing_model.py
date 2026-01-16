"""
Dynamic Pricing Optimization Model
Uses machine learning to optimize pricing based on demand, competition, and inventory
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class DynamicPricingOptimizer:
    """
    Optimizes pricing strategy using machine learning
    Considers demand elasticity, competition, and inventory levels
    """
    
    def __init__(self):
        self.price_model = None
        self.demand_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
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
        
        # Drop date column if present
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        # Store feature names
        if fit:
            self.feature_names = df.columns.tolist()
            # Remove target variables
            for col in ['optimal_price', 'demand', 'revenue', 'profit']:
                if col in self.feature_names:
                    self.feature_names.remove(col)
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the pricing optimization model"""
        print("Training Dynamic Pricing Model...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, fit=True)
        
        # Prepare features and targets
        X = df_processed.drop(['optimal_price', 'demand', 'revenue', 'profit'], axis=1)
        y_price = df_processed['optimal_price']
        y_demand = df_processed['demand']
        
        # Split data
        X_train, X_test, y_price_train, y_price_test = train_test_split(
            X, y_price, test_size=test_size, random_state=random_state
        )
        _, _, y_demand_train, y_demand_test = train_test_split(
            X, y_demand, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train price model
        print("  - Training price prediction model...")
        self.price_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        self.price_model.fit(X_train_scaled, y_price_train)
        
        # Evaluate price model
        y_price_pred = self.price_model.predict(X_test_scaled)
        price_rmse = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
        price_r2 = r2_score(y_price_test, y_price_pred)
        price_mae = mean_absolute_error(y_price_test, y_price_pred)
        
        print(f"    Price Model - RMSE: {price_rmse:.2f}, R²: {price_r2:.4f}, MAE: {price_mae:.2f}")
        
        # Train demand model
        print("  - Training demand prediction model...")
        self.demand_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        self.demand_model.fit(X_train_scaled, y_demand_train)
        
        # Evaluate demand model
        y_demand_pred = self.demand_model.predict(X_test_scaled)
        demand_rmse = np.sqrt(mean_squared_error(y_demand_test, y_demand_pred))
        demand_r2 = r2_score(y_demand_test, y_demand_pred)
        demand_mae = mean_absolute_error(y_demand_test, y_demand_pred)
        
        print(f"    Demand Model - RMSE: {demand_rmse:.2f}, R²: {demand_r2:.4f}, MAE: {demand_mae:.2f}")
        
        return {
            'price_model': {'rmse': price_rmse, 'r2': price_r2, 'mae': price_mae},
            'demand_model': {'rmse': demand_rmse, 'r2': demand_r2, 'mae': demand_mae}
        }
    
    def predict_optimal_price(self, features_df):
        """Predict optimal price for given features"""
        features_processed = self.preprocess_data(features_df, fit=False)
        X = features_processed.drop(['optimal_price', 'demand', 'revenue', 'profit'], 
                                   axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        predicted_price = self.price_model.predict(X_scaled)
        predicted_demand = self.demand_model.predict(X_scaled)
        
        return predicted_price, predicted_demand
    
    def optimize_pricing_strategy(self, features_df):
        """
        Optimize pricing strategy considering:
        - Demand elasticity
        - Inventory levels
        - Competition
        - Profit maximization
        """
        features_processed = self.preprocess_data(features_df, fit=False)
        X = features_processed.drop(['optimal_price', 'demand', 'revenue', 'profit'], 
                                   axis=1, errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predicted_prices = self.price_model.predict(X_scaled)
        predicted_demands = self.demand_model.predict(X_scaled)
        
        # Adjust prices based on inventory levels
        inventory_levels = features_df['inventory_level'].values
        production_costs = features_df['production_cost'].values
        
        optimized_prices = predicted_prices.copy()
        
        # High inventory: reduce prices to increase demand
        high_inventory = inventory_levels > 500
        optimized_prices[high_inventory] *= 0.95
        
        # Low inventory: increase prices to maximize profit
        low_inventory = inventory_levels < 100
        optimized_prices[low_inventory] *= 1.05
        
        # Ensure prices are above production cost
        optimized_prices = np.maximum(optimized_prices, production_costs * 1.2)
        
        return optimized_prices, predicted_demands
    
    def save_model(self, model_path='models'):
        """Save trained models"""
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.price_model, f'{model_path}/price_model.pkl')
        joblib.dump(self.demand_model, f'{model_path}/demand_model.pkl')
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_path}/label_encoders.pkl')
        print(f"✓ Models saved to {model_path}/")
    
    def load_model(self, model_path='models'):
        """Load trained models"""
        self.price_model = joblib.load(f'{model_path}/price_model.pkl')
        self.demand_model = joblib.load(f'{model_path}/demand_model.pkl')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{model_path}/label_encoders.pkl')
        print(f"✓ Models loaded from {model_path}/")

def main():
    """Main execution"""
    from generate_data import generate_dynamic_pricing_data
    
    # Generate data
    print("=" * 60)
    print("DYNAMIC PRICING OPTIMIZATION")
    print("=" * 60)
    
    df = generate_dynamic_pricing_data(n_samples=1000)
    
    # Initialize and train model
    optimizer = DynamicPricingOptimizer()
    metrics = optimizer.train(df)
    
    # Save models
    optimizer.save_model()
    
    # Test optimization
    print("\n" + "=" * 60)
    print("PRICING OPTIMIZATION RESULTS")
    print("=" * 60)
    
    test_df = df.head(10).copy()
    optimized_prices, predicted_demands = optimizer.optimize_pricing_strategy(test_df)
    
    results = pd.DataFrame({
        'Original_Price': test_df['optimal_price'].values,
        'Optimized_Price': optimized_prices,
        'Predicted_Demand': predicted_demands,
        'Inventory_Level': test_df['inventory_level'].values,
        'Competitor_Price': test_df['competitor_price'].values
    })
    
    print("\nSample Optimization Results:")
    print(results.to_string(index=False))
    
    print("\n✓ Dynamic Pricing Model training completed successfully!")

if __name__ == '__main__':
    main()
