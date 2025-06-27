import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

class F1ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_names = []
        
    def load_data(self):
        """Load preprocessed F1 data"""
        print("Loading preprocessed F1 data...")
        
        X_train = np.load('X_train_f1.npy')
        X_test = np.load('X_test_f1.npy')
        y_train = np.load('y_train_f1.npy')
        y_test = np.load('y_test_f1.npy')
        
        with open('feature_names_f1.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_absolute_error', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Predictions and evaluation
        y_pred_train = best_rf.predict(X_train)
        y_pred_test = best_rf.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        metrics = {
            'model': 'Random Forest',
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'cv_score': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Random Forest - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        print("Top 5 most important features:")
        print(feature_importance.head())
        
        # Save model
        joblib.dump(best_rf, 'f1_random_forest_model.pkl')
        
        self.models['random_forest'] = best_rf
        self.metrics['random_forest'] = metrics
        
        return best_rf, metrics, feature_importance
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost-style model using GradientBoosting"""
        print("Training XGBoost-style model...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=3, scoring='neg_mean_absolute_error', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_gb = grid_search.best_estimator_
        
        # Predictions and evaluation
        y_pred_train = best_gb.predict(X_train)
        y_pred_test = best_gb.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_gb, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        metrics = {
            'model': 'XGBoost',
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'cv_score': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        print(f"XGBoost - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        # Save model
        joblib.dump(best_gb, 'f1_xgboost_model.pkl')
        
        self.models['xgboost'] = best_gb
        self.metrics['xgboost'] = metrics
        
        return best_gb, metrics
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model"""
        print("Training Neural Network model...")
        
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (128, 64, 32), (200, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000]
        }
        
        nn = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)
        grid_search = GridSearchCV(
            nn, param_grid, cv=3, scoring='neg_mean_absolute_error', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_nn = grid_search.best_estimator_
        
        # Predictions and evaluation
        y_pred_train = best_nn.predict(X_train)
        y_pred_test = best_nn.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_nn, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
        
        metrics = {
            'model': 'Neural Network',
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'cv_score': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        print(f"Neural Network - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        # Save model
        joblib.dump(best_nn, 'f1_neural_network_model.pkl')
        
        self.models['neural_network'] = best_nn
        self.metrics['neural_network'] = metrics
        
        return best_nn, metrics
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression baseline"""
        print("Training Linear Regression baseline...")
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        metrics = {
            'model': 'Linear Regression',
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'cv_score': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': {}
        }
        
        print(f"Linear Regression - Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        # Save model
        joblib.dump(lr, 'f1_linear_regression_model.pkl')
        
        self.models['linear_regression'] = lr
        self.metrics['linear_regression'] = metrics
        
        return lr, metrics
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*80)
        print("F1 MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'Train MAE':<12} {'Test MAE':<12} {'Test R²':<12} {'CV Score':<12}")
        print("-" * 80)
        
        best_model = None
        best_score = float('inf')
        
        for name, metrics in self.metrics.items():
            print(f"{metrics['model']:<20} {metrics['train_mae']:<12.3f} {metrics['test_mae']:<12.3f} "
                  f"{metrics['test_r2']:<12.3f} {metrics['cv_score']:<12.3f}")
            
            if metrics['test_mae'] < best_score:
                best_score = metrics['test_mae']
                best_model = metrics['model']
        
        print("-" * 80)
        print(f"Best Model: {best_model} (Test MAE: {best_score:.3f})")
        
        return best_model
    
    def save_results(self):
        """Save all results"""
        print("Saving model results...")
        
        # Save metrics
        with open('f1_model_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save model comparison
        comparison_df = pd.DataFrame([
            {
                'Model': metrics['model'],
                'Train_MAE': metrics['train_mae'],
                'Test_MAE': metrics['test_mae'],
                'Test_R2': metrics['test_r2'],
                'CV_Score': metrics['cv_score'],
                'CV_Std': metrics['cv_std']
            }
            for metrics in self.metrics.values()
        ])
        
        comparison_df.to_csv('f1_model_comparison.csv', index=False)
        
        print("Results saved successfully!")
    
    def train_all_models(self):
        """Train all models"""
        print("Starting F1 model training pipeline...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Train models
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Linear Regression (baseline)
        self.train_linear_regression(X_train, y_train, X_test, y_test)
        
        # Random Forest
        rf_model, rf_metrics, feature_importance = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # XGBoost
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Neural Network
        self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Compare models
        best_model = self.compare_models()
        
        # Save results
        self.save_results()
        
        print(f"\nF1 model training completed!")
        print(f"Best performing model: {best_model}")
        
        return self.models, self.metrics, feature_importance

def main():
    """Main training function"""
    trainer = F1ModelTrainer()
    models, metrics, feature_importance = trainer.train_all_models()
    
    print("\nTraining pipeline completed successfully!")
    print("Models ready for F1 race predictions!")
    
    return models, metrics, feature_importance

if __name__ == "__main__":
    main()
