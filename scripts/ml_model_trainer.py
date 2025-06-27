import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def create_training_data(self):
        """Create comprehensive F1 training dataset"""
        print("Creating F1 training dataset...")
        
        # Simulate comprehensive historical F1 data (1950-2024)
        np.random.seed(42)  # For reproducible results
        
        # Create realistic F1 historical data
        years = list(range(1950, 2025))
        drivers = [
            'Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Lando Norris',
            'Oscar Piastri', 'George Russell', 'Carlos Sainz', 'Fernando Alonso',
            'Pierre Gasly', 'Yuki Tsunoda', 'Lance Stroll', 'Nico Hulkenberg',
            'Kevin Magnussen', 'Alex Albon', 'Logan Sargeant', 'Nyck de Vries',
            'Sebastian Vettel', 'Kimi Raikkonen', 'Daniel Ricciardo', 'Valtteri Bottas'
        ]
        
        teams = [
            'Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine', 'Aston Martin',
            'Williams', 'AlphaTauri', 'Haas', 'Alfa Romeo'
        ]
        
        circuits = [
            'bahrain', 'jeddah', 'albert_park', 'suzuka', 'shanghai', 'miami',
            'imola', 'monaco', 'villeneuve', 'catalunya', 'silverstone', 'hungaroring'
        ]
        
        # Generate training data
        training_data = []
        
        for year in years:
            for circuit in circuits:
                for driver in drivers:
                    # Skip drivers who weren't active in certain years
                    if year < 2007 and driver == 'Lewis Hamilton':
                        continue
                    if year < 2015 and driver == 'Max Verstappen':
                        continue
                    if year > 2022 and driver == 'Sebastian Vettel':
                        continue
                        
                    # Create realistic race data
                    team = np.random.choice(teams)
                    
                    # Driver skill factors (realistic ratings)
                    driver_skill = {
                        'Max Verstappen': 95, 'Lewis Hamilton': 92, 'Charles Leclerc': 89,
                        'Lando Norris': 87, 'Oscar Piastri': 84, 'George Russell': 82,
                        'Carlos Sainz': 85, 'Fernando Alonso': 88, 'Pierre Gasly': 78,
                        'Yuki Tsunoda': 75, 'Lance Stroll': 70, 'Nico Hulkenberg': 76,
                        'Kevin Magnussen': 74, 'Alex Albon': 72, 'Logan Sargeant': 68,
                        'Sebastian Vettel': 90, 'Kimi Raikkonen': 86, 'Daniel Ricciardo': 83,
                        'Valtteri Bottas': 80, 'Nyck de Vries': 69
                    }.get(driver, 70)
                    
                    # Team performance factors
                    team_performance = {
                        'Red Bull': 95, 'Mercedes': 88, 'Ferrari': 87, 'McLaren': 86,
                        'Alpine': 75, 'Aston Martin': 78, 'Williams': 65, 'AlphaTauri': 72,
                        'Haas': 68, 'Alfa Romeo': 63
                    }.get(team, 70)
                    
                    # Weather conditions
                    temperature = np.random.normal(25, 8)
                    humidity = np.random.normal(60, 20)
                    rain_probability = np.random.beta(2, 8)  # Skewed towards dry conditions
                    
                    # Circuit characteristics
                    circuit_difficulty = np.random.uniform(5, 10)
                    overtaking_difficulty = np.random.uniform(3, 9)
                    
                    # Calculate realistic finishing position based on multiple factors
                    base_performance = (driver_skill * 0.4 + team_performance * 0.6) / 100
                    
                    # Weather impact
                    if rain_probability > 0.3:
                        # Some drivers are better in wet conditions
                        wet_bonus = {'Lewis Hamilton': 0.15, 'Max Verstappen': 0.12, 
                                   'Fernando Alonso': 0.14}.get(driver, 0.05)
                        base_performance += wet_bonus
                    
                    # Temperature impact
                    if temperature > 35:
                        base_performance *= 0.95  # Hot weather reliability issues
                    
                    # Circuit-specific performance
                    if circuit == 'monaco' and driver in ['Charles Leclerc', 'Max Verstappen']:
                        base_performance *= 1.1
                    
                    # Add randomness for race unpredictability
                    race_randomness = np.random.normal(0, 0.1)
                    final_performance = base_performance + race_randomness
                    
                    # Convert performance to position (1-20)
                    position = max(1, min(20, int(21 - final_performance * 20)))
                    
                    # Create feature vector
                    race_data = {
                        'driver': driver,
                        'team': team,
                        'circuit': circuit,
                        'year': year,
                        'temperature': temperature,
                        'humidity': humidity,
                        'rain_probability': rain_probability,
                        'circuit_difficulty': circuit_difficulty,
                        'overtaking_difficulty': overtaking_difficulty,
                        'driver_skill': driver_skill,
                        'team_performance': team_performance,
                        'driver_experience': min(year - 2000, 25),  # Years of experience
                        'championship_points': np.random.randint(0, 400),
                        'recent_form': np.random.uniform(1, 20),  # Average recent position
                        'grid_position': np.random.randint(1, 21),
                        'position': position  # Target variable
                    }
                    
                    training_data.append(race_data)
        
        df = pd.DataFrame(training_data)
        print(f"Created training dataset with {len(df)} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML training"""
        print("Preparing features for ML training...")
        
        # Encode categorical variables
        categorical_features = ['driver', 'team', 'circuit']
        
        for feature in categorical_features:
            le = LabelEncoder()
            df[f'{feature}_encoded'] = le.fit_transform(df[feature])
            self.encoders[feature] = le
        
        # Select features for training
        feature_columns = [
            'driver_encoded', 'team_encoded', 'circuit_encoded', 'year',
            'temperature', 'humidity', 'rain_probability', 'circuit_difficulty',
            'overtaking_difficulty', 'driver_skill', 'team_performance',
            'driver_experience', 'championship_points', 'recent_form', 'grid_position'
        ]
        
        X = df[feature_columns].copy()
        y = df['position'].copy()
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = [
            'year', 'temperature', 'humidity', 'rain_probability', 'circuit_difficulty',
            'overtaking_difficulty', 'driver_skill', 'team_performance',
            'driver_experience', 'championship_points', 'recent_form', 'grid_position'
        ]
        
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        self.scalers['feature_scaler'] = scaler
        self.feature_names = feature_columns
        
        print(f"Prepared {len(feature_columns)} features for training")
        return X, y
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"Random Forest - MAE: {rf_mae:.3f}, R²: {rf_r2:.3f}")
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        # Evaluate XGBoost
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        print(f"XGBoost - MAE: {xgb_mae:.3f}, R²: {xgb_r2:.3f}")
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Evaluate Gradient Boosting
        gb_pred = gb_model.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        
        print(f"Gradient Boosting - MAE: {gb_mae:.3f}, R²: {gb_r2:.3f}")
        
        # Store models
        self.models = {
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'gradient_boosting': gb_model
        }
        
        # Store metrics
        self.metrics = {
            'random_forest': {'mae': rf_mae, 'r2': rf_r2},
            'xgboost': {'mae': xgb_mae, 'r2': xgb_r2},
            'gradient_boosting': {'mae': gb_mae, 'r2': gb_r2}
        }
        
        # Select best model
        best_model_name = min(self.metrics.keys(), key=lambda x: self.metrics[x]['mae'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name} (MAE: {self.metrics[best_model_name]['mae']:.3f})")
        
        return X_test, y_test
    
    def save_models(self):
        """Save trained models and preprocessors"""
        print("Saving trained models...")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'f1_model_{name}.pkl')
        
        # Save preprocessors
        joblib.dump(self.scalers, 'f1_scalers.pkl')
        joblib.dump(self.encoders, 'f1_encoders.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_model': self.best_model_name,
            'metrics': self.metrics,
            'training_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        with open('f1_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Models saved successfully!")
    
    def train_complete_pipeline(self):
        """Complete ML training pipeline"""
        print("Starting F1 ML Model Training Pipeline...")
        print("=" * 60)
        
        # Create training data
        df = self.create_training_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train models
        X_test, y_test = self.train_models(X, y)
        
        # Save models
        self.save_models()
        
        self.is_trained = True
        
        print("=" * 60)
        print("F1 ML Model Training Complete!")
        print(f"Best Model: {self.best_model_name}")
        print(f"Test MAE: {self.metrics[self.best_model_name]['mae']:.3f}")
        print(f"Test R²: {self.metrics[self.best_model_name]['r2']:.3f}")
        
        return self.models, self.metrics

def main():
    """Main training function"""
    trainer = F1MLModelTrainer()
    models, metrics = trainer.train_complete_pipeline()
    
    print("\n🏁 F1 ML Models Ready for Predictions! 🏁")
    return trainer

if __name__ == "__main__":
    trainer = main()
