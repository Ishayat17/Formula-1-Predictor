import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json

def load_training_data():
    """Load preprocessed training data"""
    print("Loading training data...")
    
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    
    # Predictions and evaluation
    y_pred = best_rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    metrics = {
        'model': 'Random Forest',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'cv_score': -cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }
    
    # Save model
    joblib.dump(best_rf, 'random_forest_model.pkl')
    
    print(f"Random Forest - MAE: {mae:.3f}, R²: {r2:.3f}")
    return best_rf, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("Training XGBoost model...")
    
    # Using GradientBoostingRegressor as XGBoost alternative
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_gb = grid_search.best_estimator_
    
    # Predictions and evaluation
    y_pred = best_gb.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(best_gb, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    metrics = {
        'model': 'XGBoost',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'cv_score': -cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }
    
    # Save model
    joblib.dump(best_gb, 'xgboost_model.pkl')
    
    print(f"XGBoost - MAE: {mae:.3f}, R²: {r2:.3f}")
    return best_gb, metrics

def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Neural Network model"""
    print("Training Neural Network model...")
    
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (128, 64, 32)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    nn = MLPRegressor(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_nn = grid_search.best_estimator_
    
    # Predictions and evaluation
    y_pred = best_nn.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(best_nn, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    
    metrics = {
        'model': 'Neural Network',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'cv_score': -cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }
    
    # Save model
    joblib.dump(best_nn, 'neural_network_model.pkl')
    
    print(f"Neural Network - MAE: {mae:.3f}, R²: {r2:.3f}")
    return best_nn, metrics

def compare_models(metrics_list):
    """Compare model performance"""
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<15} {'MAE':<8} {'R²':<8} {'CV Score':<10}")
    print("-" * 60)
    
    for metrics in metrics_list:
        print(f"{metrics['model']:<15} {metrics['mae']:<8.3f} {metrics['r2']:<8.3f} {metrics['cv_score']:<10.3f}")
    
    # Find best model
    best_model = min(metrics_list, key=lambda x: x['mae'])
    print(f"\nBest Model: {best_model['model']} (MAE: {best_model['mae']:.3f})")
    
    return best_model

def main():
    """Main training pipeline"""
    print("Starting F1 model training pipeline...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Train models
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    nn_model, nn_metrics = train_neural_network(X_train, y_train, X_test, y_test)
    
    # Compare models
    all_metrics = [rf_metrics, xgb_metrics, nn_metrics]
    best_model = compare_models(all_metrics)
    
    # Save metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nTraining completed! Best model: {best_model['model']}")
    print("All models and metrics saved successfully.")

if __name__ == "__main__":
    main()
