#!/usr/bin/env python3
"""
Improved F1 Model Training Script
Retrains models with hyperparameter tuning, cross-validation, and comprehensive evaluation
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings('ignore')

class ImprovedF1ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        self.best_params = {}
        self.cv_scores = {}
        
    def load_and_clean_data(self):
        """Load and clean F1 data"""
        print("🔄 Loading and cleaning F1 data...")
        
        try:
            # Load core datasets
            results = pd.read_csv('data/results.csv')
            races = pd.read_csv('data/races.csv')
            circuits = pd.read_csv('data/circuits.csv')
            
            print(f"✓ Loaded {len(results)} race results")
            print(f"✓ Loaded {len(races)} races")
            print(f"✓ Loaded {len(circuits)} circuits")
            
            # Merge datasets
            df = results.merge(races, on='raceId', how='left')
            df = df.merge(circuits, on='circuitId', how='left')
            
            # Clean numeric columns
            df['position'] = pd.to_numeric(df['position'], errors='coerce')
            df['position'] = df['position'].fillna(21)  # DNF = 21st
            
            df['points'] = pd.to_numeric(df['points'], errors='coerce')
            df['points'] = df['points'].fillna(0)
            
            df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
            df['grid'] = df['grid'].fillna(20)
            
            # Clean year and round
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['round'] = pd.to_numeric(df['round'], errors='coerce')
            
            # Handle missing values
            df = df.dropna(subset=['driverId', 'constructorId', 'circuitId'])
            
            # Split data: use pre-2004 for training, 2004+ for prediction
            df_train = df[df['year'] < 2004].copy()
            df_predict = df[df['year'] >= 2004].copy()
            
            print(f"✓ Training data (pre-2004): {len(df_train)} records")
            print(f"✓ Prediction data (2004+): {len(df_predict)} records")
            
            return df_train, df_predict
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return self._create_mock_data(), self._create_mock_data()
    
    def _create_mock_data(self):
        """Create realistic mock data when real data fails"""
        print("🔄 Creating mock data...")
        
        np.random.seed(42)
        n_records = 5000
        
        data = {
            'driverId': np.random.randint(1, 50, n_records),
            'constructorId': np.random.randint(1, 20, n_records),
            'circuitId': np.random.randint(1, 30, n_records),
            'year': np.random.randint(1950, 2004, n_records),
            'round': np.random.randint(1, 25, n_records),
            'position': np.random.randint(1, 21, n_records),
            'points': np.random.randint(0, 26, n_records),
            'grid': np.random.randint(1, 21, n_records),
            'lat': np.random.uniform(20, 60, n_records),
            'lng': np.random.uniform(-10, 50, n_records),
            'alt': np.random.uniform(0, 1000, n_records)
        }
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """Engineer features for prediction"""
        print("🔧 Engineering features...")
        
        # Sort by driver, year, round for rolling calculations
        df_sorted = df.sort_values(['driverId', 'year', 'round'])
        
        # 1. Driver Performance Features
        df['driver_avg_position_5races'] = df_sorted.groupby('driverId')['position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(10.5)
        
        df['driver_avg_position_10races'] = df_sorted.groupby('driverId')['position'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
        ).fillna(10.5)
        
        df['driver_recent_form'] = df_sorted.groupby('driverId')['position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        ).fillna(10.5)
        
        # 2. Constructor Performance Features
        df['constructor_avg_position_5races'] = df_sorted.groupby('constructorId')['position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(10.5)
        
        df['constructor_avg_points_5races'] = df_sorted.groupby('constructorId')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(5.0)
        
        # 3. Circuit Features
        df['driver_circuit_avg'] = df_sorted.groupby(['driverId', 'circuitId'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(10.5)
        
        df['circuit_difficulty'] = df.groupby('circuitId')['position'].transform('std').fillna(5.0)
        
        # 4. Seasonal Features
        df['season_race_number'] = df.groupby(['driverId', 'year']).cumcount() + 1
        df['season_points_so_far'] = df_sorted.groupby(['driverId', 'year'])['points'].cumsum().shift(1).fillna(0)
        
        # 5. Grid and Position Features
        df['grid_advantage'] = df['position'] - df['grid']
        
        # 6. Basic Features
        df['year_normalized'] = (df['year'] - 1950) / 100
        df['round_normalized'] = df['round'] / 25
        
        print("✓ Feature engineering completed")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("🔧 Encoding categorical features...")
        
        categorical_features = ['driverId', 'constructorId', 'circuitId']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.encoders[feature] = le
        
        return df
    
    def prepare_features(self, df):
        """Prepare final feature set"""
        print("🔧 Preparing final features...")
        
        # Select features
        feature_columns = [
            'grid', 'year_normalized', 'round_normalized',
            'lat', 'lng', 'alt',
            'driver_avg_position_5races', 'driver_avg_position_10races',
            'driver_recent_form', 'driver_circuit_avg',
            'constructor_avg_position_5races', 'constructor_avg_points_5races',
            'circuit_difficulty', 'season_race_number', 'season_points_so_far',
            'grid_advantage',
            'driverId_encoded', 'constructorId_encoded', 'circuitId_encoded'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Target variable
        y = df['position'].copy()
        
        print(f"✓ Final feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        self.feature_names = list(X.columns)
        
        return X, y
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Position accuracy metrics
        position_accuracy_1 = np.mean(np.abs(y_pred - y_true) <= 1) * 100
        position_accuracy_2 = np.mean(np.abs(y_pred - y_true) <= 2) * 100
        position_accuracy_3 = np.mean(np.abs(y_pred - y_true) <= 3) * 100
        
        # Classification accuracy metrics
        podium_accuracy = np.mean((y_pred <= 3) == (y_true <= 3)) * 100
        top5_accuracy = np.mean((y_pred <= 5) == (y_true <= 5)) * 100
        top10_accuracy = np.mean((y_pred <= 10) == (y_true <= 10)) * 100
        
        # Direction accuracy
        direction_accuracy = np.mean((y_pred > 10) == (y_true > 10)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'position_accuracy_1': position_accuracy_1,
            'position_accuracy_2': position_accuracy_2,
            'position_accuracy_3': position_accuracy_3,
            'podium_accuracy': podium_accuracy,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'direction_accuracy': direction_accuracy
        }
    
    def train_models(self, X, y):
        """Train models with hyperparameter tuning and cross-validation"""
        print("🤖 Training models with hyperparameter tuning...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define hyperparameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        
        # Initialize models
        base_models = {
            'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'xgboost': GradientBoostingRegressor(random_state=42),
            'neural_network': MLPRegressor(random_state=42, max_iter=1000)
        }
        
        results = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in base_models.items():
            print(f"  Training {name} with hyperparameter tuning...")
            
            # Perform cross-validation
            if name == 'neural_network':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='neg_mean_absolute_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
            
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"    Cross-validation MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
            
            # Hyperparameter tuning
            if name == 'neural_network':
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params[name] = grid_search.best_params_
            
            print(f"    Best parameters: {grid_search.best_params_}")
            print(f"    Best CV score: {-grid_search.best_score_:.3f}")
            
            # Evaluate on test set
            if name == 'neural_network':
                y_pred = best_model.predict(X_test_scaled)
            else:
                y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Feature importance
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, best_model.feature_importances_))
            
            results[name] = {
                'model': best_model,
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'best_params': grid_search.best_params_,
                'feature_importance': feature_importance,
                **metrics
            }
            
            print(f"    Test MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
            print(f"    Position Accuracy (±1): {metrics['position_accuracy_1']:.1f}%")
            print(f"    Podium Accuracy: {metrics['podium_accuracy']:.1f}%")
            print(f"    Top 10 Accuracy: {metrics['top10_accuracy']:.1f}%")
        
        # Save models and metadata
        for name, result in results.items():
            joblib.dump(result['model'], f'improved_{name}_model.pkl')
        
        joblib.dump(self.scaler, 'improved_scaler.pkl')
        joblib.dump(self.encoders, 'improved_encoders.pkl')
        joblib.dump(self.best_params, 'improved_best_params.pkl')
        
        with open('improved_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save comprehensive results
        results_for_save = {}
        for name, result in results.items():
            results_for_save[name] = {
                'mae': result['mae'],
                'mse': result['mse'],
                'rmse': result['rmse'],
                'r2': result['r2'],
                'cv_mae': result['cv_mae'],
                'cv_std': result['cv_std'],
                'position_accuracy_1': result['position_accuracy_1'],
                'position_accuracy_2': result['position_accuracy_2'],
                'position_accuracy_3': result['position_accuracy_3'],
                'podium_accuracy': result['podium_accuracy'],
                'top5_accuracy': result['top5_accuracy'],
                'top10_accuracy': result['top10_accuracy'],
                'direction_accuracy': result['direction_accuracy'],
                'best_params': result['best_params'],
                'feature_importance': result['feature_importance']
            }
        
        with open('improved_model_results.json', 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        self.models = {name: result['model'] for name, result in results.items()}
        
        print("✓ Models trained and saved successfully!")
        return results
    
    def train_all(self):
        """Complete training pipeline"""
        print("🚀 Starting improved F1 model training...")
        
        # Load and clean data
        df_train, df_predict = self.load_and_clean_data()
        
        # Engineer features
        df_train = self.engineer_features(df_train)
        df_predict = self.engineer_features(df_predict)
        
        # Encode categorical features
        df_train = self.encode_categorical_features(df_train)
        df_predict = self.encode_categorical_features(df_predict)
        
        # Prepare features
        X_train, y_train = self.prepare_features(df_train)
        X_predict, y_predict = self.prepare_features(df_predict)
        
        # Train models
        results = self.train_models(X_train, y_train)
        
        print("✅ Improved F1 model training completed!")
        return results

def main():
    """Main function"""
    trainer = ImprovedF1ModelTrainer()
    
    # Train models
    results = trainer.train_all()
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  MAE: {result['mae']:.3f}")
        print(f"  R²: {result['r2']:.3f}")
        print(f"  Position Accuracy (±1): {result['position_accuracy_1']:.1f}%")
        print(f"  Podium Accuracy: {result['podium_accuracy']:.1f}%")
        print(f"  Top 10 Accuracy: {result['top10_accuracy']:.1f}%")
        
        if result['feature_importance']:
            print(f"  Top 5 Features:")
            sorted_features = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"    {feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 