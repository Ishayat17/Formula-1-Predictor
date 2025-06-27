import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WorldClassF1ModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_names = []
        self.ensemble_model = None
        
    def load_world_class_data(self):
        """Load world-class preprocessed F1 data"""
        print("Loading world-class F1 data...")
        
        X_train = np.load('X_train_world_class.npy')
        X_val = np.load('X_val_world_class.npy')
        X_test = np.load('X_test_world_class.npy')
        y_train = np.load('y_train_world_class.npy')
        y_val = np.load('y_val_world_class.npy')
        y_test = np.load('y_test_world_class.npy')
        
        with open('world_class_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
        
        print(f"Data loaded successfully:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples") 
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {len(self.feature_names)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with advanced hyperparameter tuning"""
        print("Training XGBoost model...")
        
        # Advanced parameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # Use RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=50, cv=5, 
            scoring='neg_mean_absolute_error', n_jobs=-1, 
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_xgb = random_search.best_estimator_
        
        # Evaluate
        y_pred_train = best_xgb.predict(X_train)
        y_pred_val = best_xgb.predict(X_val)
        
        metrics = self._calculate_metrics('XGBoost', y_train, y_pred_train, y_val, y_pred_val)
        metrics['best_params'] = random_search.best_params_
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"XGBoost - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        
        joblib.dump(best_xgb, 'world_class_xgboost_model.pkl')
        self.models['xgboost'] = best_xgb
        self.metrics['xgboost'] = metrics
        
        return best_xgb, metrics, feature_importance
    
    def train_lightgbm_model(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
            'min_child_samples': [10, 20, 30]
        }
        
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=50, cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1,
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_lgb = random_search.best_estimator_
        
        y_pred_train = best_lgb.predict(X_train)
        y_pred_val = best_lgb.predict(X_val)
        
        metrics = self._calculate_metrics('LightGBM', y_train, y_pred_train, y_val, y_pred_val)
        metrics['best_params'] = random_search.best_params_
        
        print(f"LightGBM - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        
        joblib.dump(best_lgb, 'world_class_lightgbm_model.pkl')
        self.models['lightgbm'] = best_lgb
        self.metrics['lightgbm'] = metrics
        
        return best_lgb, metrics
    
    def train_advanced_random_forest(self, X_train, y_train, X_val, y_val):
        """Train advanced Random Forest with extensive tuning"""
        print("Training Advanced Random Forest...")
        
        param_grid = {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
            'max_samples': [0.8, 0.9, 1.0]
        }
        
        rf_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        random_search = RandomizedSearchCV(
            rf_model, param_grid, n_iter=100, cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1,
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_rf = random_search.best_estimator_
        
        y_pred_train = best_rf.predict(X_train)
        y_pred_val = best_rf.predict(X_val)
        
        metrics = self._calculate_metrics('Random Forest', y_train, y_pred_train, y_val, y_pred_val)
        metrics['best_params'] = random_search.best_params_
        metrics['oob_score'] = best_rf.oob_score_
        
        print(f"Random Forest - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        print(f"OOB Score: {metrics['oob_score']:.3f}")
        
        joblib.dump(best_rf, 'world_class_random_forest_model.pkl')
        self.models['random_forest'] = best_rf
        self.metrics['random_forest'] = metrics
        
        return best_rf, metrics
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train advanced neural network"""
        print("Training Advanced Neural Network...")
        
        param_grid = {
            'hidden_layer_sizes': [
                (100,), (200,), (300,),
                (100, 50), (200, 100), (300, 150),
                (100, 50, 25), (200, 100, 50), (300, 150, 75),
                (500, 250, 125, 60)
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [500, 1000, 1500],
            'early_stopping': [True],
            'validation_fraction': [0.1, 0.15, 0.2]
        }
        
        nn_model = MLPRegressor(
            random_state=42,
            solver='adam'
        )
        
        random_search = RandomizedSearchCV(
            nn_model, param_grid, n_iter=50, cv=3,
            scoring='neg_mean_absolute_error', n_jobs=-1,
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_nn = random_search.best_estimator_
        
        y_pred_train = best_nn.predict(X_train)
        y_pred_val = best_nn.predict(X_val)
        
        metrics = self._calculate_metrics('Neural Network', y_train, y_pred_train, y_val, y_pred_val)
        metrics['best_params'] = random_search.best_params_
        
        print(f"Neural Network - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        
        joblib.dump(best_nn, 'world_class_neural_network_model.pkl')
        self.models['neural_network'] = best_nn
        self.metrics['neural_network'] = metrics
        
        return best_nn, metrics
    
    def train_support_vector_regression(self, X_train, y_train, X_val, y_val):
        """Train SVR model"""
        print("Training Support Vector Regression...")
        
        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        
        svr_model = SVR()
        
        random_search = RandomizedSearchCV(
            svr_model, param_grid, n_iter=50, cv=3,
            scoring='neg_mean_absolute_error', n_jobs=-1,
            random_state=42, verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_svr = random_search.best_estimator_
        
        y_pred_train = best_svr.predict(X_train)
        y_pred_val = best_svr.predict(X_val)
        
        metrics = self._calculate_metrics('SVR', y_train, y_pred_train, y_val, y_pred_val)
        metrics['best_params'] = random_search.best_params_
        
        print(f"SVR - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        
        joblib.dump(best_svr, 'world_class_svr_model.pkl')
        self.models['svr'] = best_svr
        self.metrics['svr'] = metrics
        
        return best_svr, metrics
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Create advanced ensemble model"""
        print("Creating World-Class Ensemble Model...")
        
        # Select best performing models for ensemble
        best_models = []
        model_names = []
        
        for name, metrics in self.metrics.items():
            if metrics['val_mae'] < 3.0:  # Only include models with good performance
                best_models.append((name, self.models[name]))
                model_names.append(name)
        
        print(f"Selected {len(best_models)} models for ensemble: {model_names}")
        
        # Create stacking ensemble
        meta_learner = Ridge(alpha=1.0)
        
        stacking_regressor = StackingRegressor(
            estimators=best_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        stacking_regressor.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_train = stacking_regressor.predict(X_train)
        y_pred_val = stacking_regressor.predict(X_val)
        
        metrics = self._calculate_metrics('Ensemble', y_train, y_pred_train, y_val, y_pred_val)
        
        print(f"Ensemble - Val MAE: {metrics['val_mae']:.3f}, Val R²: {metrics['val_r2']:.3f}")
        
        joblib.dump(stacking_regressor, 'world_class_ensemble_model.pkl')
        self.ensemble_model = stacking_regressor
        self.models['ensemble'] = stacking_regressor
        self.metrics['ensemble'] = metrics
        
        return stacking_regressor, metrics
    
    def _calculate_metrics(self, model_name, y_train, y_pred_train, y_val, y_pred_val):
        """Calculate comprehensive metrics"""
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
        
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_mse = mean_squared_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        val_mape = mean_absolute_percentage_error(y_val, y_pred_val)
        
        return {
            'model': model_name,
            'train_mae': train_mae,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'val_mape': val_mape,
            'overfitting': train_mae - val_mae
        }
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Final evaluation on test set"""
        print("Evaluating models on test set...")
        
        test_results = {}
        
        for name, model in self.models.items():
            y_pred_test = model.predict(X_test)
            
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
            
            test_results[name] = {
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'test_mape': test_mape
            }
            
            print(f"{name} - Test MAE: {test_mae:.3f}, Test R²: {test_r2:.3f}")
        
        return test_results
    
    def compare_all_models(self):
        """Compare all trained models"""
        print("\n" + "="*100)
        print("WORLD-CLASS F1 MODEL COMPARISON")
        print("="*100)
        print(f"{'Model':<20} {'Train MAE':<12} {'Val MAE':<12} {'Val R²':<12} {'Val MAPE':<12} {'Overfitting':<12}")
        print("-" * 100)
        
        best_model = None
        best_score = float('inf')
        
        for name, metrics in self.metrics.items():
            overfitting = metrics.get('overfitting', 0)
            print(f"{metrics['model']:<20} {metrics['train_mae']:<12.3f} {metrics['val_mae']:<12.3f} "
                  f"{metrics['val_r2']:<12.3f} {metrics['val_mape']:<12.3f} {overfitting:<12.3f}")
            
            # Consider both accuracy and overfitting for best model selection
            combined_score = metrics['val_mae'] + abs(overfitting) * 0.5
            if combined_score < best_score:
                best_score = combined_score
                best_model = metrics['model']
        
        print("-" * 100)
        print(f"Best Model: {best_model} (Combined Score: {best_score:.3f})")
        
        return best_model
    
    def save_all_results(self):
        """Save comprehensive results"""
        print("Saving world-class model results...")
        
        # Save metrics
        with open('world_class_model_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save model comparison
        comparison_df = pd.DataFrame([
            {
                'Model': metrics['model'],
                'Train_MAE': metrics['train_mae'],
                'Val_MAE': metrics['val_mae'],
                'Val_R2': metrics['val_r2'],
                'Val_MAPE': metrics['val_mape'],
                'Overfitting': metrics.get('overfitting', 0)
            }
            for metrics in self.metrics.values()
        ])
        
        comparison_df.to_csv('world_class_model_comparison.csv', index=False)
        
        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'best_model': min(self.metrics.items(), key=lambda x: x[1]['val_mae'])[0],
            'feature_count': len(self.feature_names),
            'ensemble_available': 'ensemble' in self.models
        }
        
        with open('world_class_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("All results saved successfully!")
    
    def train_all_world_class_models(self):
        """Train all world-class models"""
        print("Starting World-Class F1 Model Training Pipeline...")
        print("="*80)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_world_class_data()
        
        # Train individual models
        print("\nTraining Individual Models...")
        print("-" * 50)
        
        # XGBoost
        self.train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # LightGBM
        self.train_lightgbm_model(X_train, y_train, X_val, y_val)
        
        # Advanced Random Forest
        self.train_advanced_random_forest(X_train, y_train, X_val, y_val)
        
        # Neural Network
        self.train_neural_network(X_train, y_train, X_val, y_val)
        
        # SVR
        self.train_support_vector_regression(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        print("\nCreating Ensemble Model...")
        print("-" * 50)
        self.create_ensemble_model(X_train, y_train, X_val, y_val)
        
        # Compare models
        best_model = self.compare_all_models()
        
        # Test set evaluation
        print("\nFinal Test Set Evaluation...")
        print("-" * 50)
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Save results
        self.save_all_results()
        
        print(f"\nWorld-Class F1 Model Training Completed!")
        print(f"Best Model: {best_model}")
        print(f"Models Available: {list(self.models.keys())}")
        
        return self.models, self.metrics, test_results

def main():
    """Main training function"""
    trainer = WorldClassF1ModelTrainer()
    models, metrics, test_results = trainer.train_all_world_class_models()
    
    print("\n🏆 WORLD-CLASS F1 PREDICTOR READY! 🏆")
    print("All models trained and ready for predictions!")
    
    return models, metrics, test_results

if __name__ == "__main__":
    main()
