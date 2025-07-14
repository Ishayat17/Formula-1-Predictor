import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import sys
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class RobustF1Predictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        self.best_params = {}
        self.cv_scores = {}
        
    def load_and_clean_data(self):
        """Load and clean F1 data robustly"""
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
            'year': np.random.randint(2010, 2025, n_records),
            'round': np.random.randint(1, 25, n_records),
            'position': np.random.randint(1, 21, n_records),
            'points': np.random.randint(0, 26, n_records),
            'grid': np.random.randint(1, 21, n_records),
            'lat': np.random.uniform(20, 60, n_records),
            'lng': np.random.uniform(-10, 50, n_records),
            'alt': np.random.uniform(0, 1000, n_records)
        }
        
        return pd.DataFrame(data)
    
    def engineer_robust_features(self, df):
        """Engineer robust features for prediction"""
        print("🔧 Engineering robust features...")
        
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
        
        # Select robust features
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
    
    def calculate_advanced_metrics(self, y_true, y_pred):
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
        
        # Direction accuracy (predicting if driver finishes better/worse than grid)
        grid_advantage_true = y_true - 10  # Assuming grid position around 10
        grid_advantage_pred = y_pred - 10
        direction_accuracy = np.mean((grid_advantage_pred > 0) == (grid_advantage_true > 0)) * 100
        
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
    
    def train_robust_models(self, X, y):
        """Train robust ML models with hyperparameter tuning and cross-validation"""
        print("🤖 Training robust ML models with hyperparameter tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define hyperparameter grids for each model
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
            
            # Hyperparameter tuning with GridSearchCV
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
            
            # Calculate comprehensive metrics
            metrics = self.calculate_advanced_metrics(y_test, y_pred)
            
            # Feature importance (for tree-based models)
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
            joblib.dump(result['model'], f'robust_{name}_model.pkl')
        
        joblib.dump(self.scaler, 'robust_scaler.pkl')
        joblib.dump(self.encoders, 'robust_encoders.pkl')
        joblib.dump(self.best_params, 'robust_best_params.pkl')
        
        with open('robust_feature_names.json', 'w') as f:
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
        
        with open('robust_model_results.json', 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.is_trained = True
        
        print("✓ Models trained and saved successfully!")
        return results
    
    def predict_race_with_real_data(self, race_conditions):
        """Predict race results using real historical data for each driver on the actual grid"""
        print(f"🏁 Predicting {race_conditions.get('race', 'Unknown')} with real historical data...")
        try:
            # Load all necessary data
            results = pd.read_csv('data/results.csv')
            races = pd.read_csv('data/races.csv')
            circuits = pd.read_csv('data/circuits.csv')
            drivers = pd.read_csv('data/drivers.csv')
            constructors = pd.read_csv('data/constructors.csv')
            
            # Filter to 2004+ data for prediction (recent F1 era)
            races_recent = races[races['year'] >= 2004]
            results_recent = results[results['raceId'].isin(races_recent['raceId'])]
            results_recent = results_recent.merge(races_recent[['raceId', 'circuitId', 'year', 'round']], on='raceId', how='left')
            
            # Find the requested race in recent data
            race_name = race_conditions.get('race', '').lower()
            year = race_conditions.get('year', None)
            if year is not None:
                race_row = races_recent[(races_recent['name'].str.lower() == race_name) & (races_recent['year'] == year)]
            else:
                race_row = races_recent[races_recent['name'].str.lower() == race_name]
            if race_row.empty:
                # fallback: try partial match
                race_row = races_recent[races_recent['name'].str.lower().str.contains(race_name)]
            if race_row.empty:
                print(f"Race '{race_conditions.get('race')}' not found in recent data (2004+)")
                return self._fallback_prediction(race_conditions)
            race_row = race_row.iloc[0]
            race_id = race_row['raceId']
            year = race_row['year']
            round_num = race_row['round']
            # Robust circuitId extraction
            circuit_id = race_row['circuitId'] if 'circuitId' in race_row else None
            if circuit_id is None or pd.isna(circuit_id):
                # Try to get circuitId from results for this race
                grid_results_tmp = results_recent[results_recent['raceId'] == race_id]
                if not grid_results_tmp.empty and 'circuitId' in grid_results_tmp.columns:
                    circuit_id = grid_results_tmp['circuitId'].iloc[0]
                else:
                    print(f"Could not determine circuitId for raceId {race_id}")
                    return self._fallback_prediction(race_conditions)
            # Get circuit info
            circuit_row = circuits[circuits['circuitId'] == circuit_id]
            if not circuit_row.empty:
                circuit_row = circuit_row.iloc[0]
                lat = circuit_row['lat']
                lng = circuit_row['lng']
                alt = circuit_row['alt']
            else:
                lat, lng, alt = 26.0, 50.0, 0.0
            # Get the real grid for this race
            grid_results = results_recent[results_recent['raceId'] == race_id]
            # Only consider drivers who actually started (grid > 0 or positionOrder > 0)
            grid_results = grid_results[(grid_results['grid'] > 0) | (grid_results['positionOrder'] > 0)]
            # Build driver and constructor lookup dicts
            driver_name_map = {row['driverId']: f"{row['forename']} {row['surname']}" for _, row in drivers.iterrows()}
            constructor_name_map = {row['constructorId']: row['name'] for _, row in constructors.iterrows()}
            # For each driver on the grid, build features using only data up to this race
            predictions = []
            for _, row in grid_results.iterrows():
                driver_id = row['driverId']
                constructor_id = row['constructorId']
                grid_pos = row['grid']
                # Use all previous races for this driver up to this race (2004+ only)
                prev_races = races_recent[(races_recent['year'] < year) | ((races_recent['year'] == year) & (races_recent['round'] < round_num))]
                prev_race_ids = prev_races['raceId'].tolist()
                driver_hist = results_recent[(results_recent['driverId'] == driver_id) & (results_recent['raceId'].isin(prev_race_ids))]
                team_hist = results_recent[(results_recent['constructorId'] == constructor_id) & (results_recent['raceId'].isin(prev_race_ids))]
                driver_circuit_hist = driver_hist[driver_hist['circuitId'] == circuit_id]
                # Calculate features
                def safe_mean(series, default):
                    return series.mean() if not series.empty else default
                driver_avg_5 = safe_mean(driver_hist['position'].tail(5), 10.5)
                driver_avg_10 = safe_mean(driver_hist['position'].tail(10), 10.5)
                driver_recent_form = safe_mean(driver_hist['position'].tail(3), 10.5)
                driver_circuit_avg = safe_mean(driver_circuit_hist['position'], 10.5)
                team_avg_5 = safe_mean(team_hist['position'].tail(5), 10.5)
                team_points_5 = safe_mean(team_hist['points'].tail(5), 5.0)
                circuit_difficulty = safe_mean(results_recent[results_recent['circuitId'] == circuit_id]['position'].std(), 7.0)
                season_race_number = prev_races[prev_races['year'] == year].shape[0] + 1
                season_points_so_far = driver_hist[driver_hist['year'] == year]['points'].sum() if not driver_hist.empty else 0.0
                grid_advantage = safe_mean(driver_hist['position'] - driver_hist['grid'], -2.0)
                # Encoded IDs
                driverId_encoded = driver_id
                constructorId_encoded = constructor_id
                circuitId_encoded = circuit_id
                # Feature vector
                features = [
                    grid_pos,
                    (year - 1950) / 100,
                    round_num / 25,
                    lat,
                    lng,
                    alt,
                    driver_avg_5,
                    driver_avg_10,
                    driver_recent_form,
                    driver_circuit_avg,
                    team_avg_5,
                    team_points_5,
                    circuit_difficulty,
                    season_race_number,
                    season_points_so_far,
                    grid_advantage,
                    driverId_encoded,
                    constructorId_encoded,
                    circuitId_encoded
                ]
                # Ensure correct number of features
                while len(features) < 19:
                    features.append(0.0)
                features = features[:19]
                # Get prediction from models
                model_predictions = {}
                for name, model in self.models.items():
                    if name == 'neural_network':
                        features_scaled = self.scaler.transform([features])
                        pred = model.predict(features_scaled)[0]
                    else:
                        pred = model.predict([features])[0]
                    model_predictions[name] = max(1, min(20, round(pred)))
                # Ensemble prediction
                weights = {'random_forest': 0.4, 'xgboost': 0.4, 'neural_network': 0.2}
                ensemble_pred = sum(model_predictions[name] * weights.get(name, 0.33) for name in model_predictions)
                ensemble_pred = max(1, min(20, round(ensemble_pred)))
                # Confidence: based on std of last 5 positions
                driver_std_5 = driver_hist['position'].tail(5).std() if not driver_hist['position'].tail(5).empty else 5.0
                confidence = max(60.0, 85.0 - (driver_std_5 if not np.isnan(driver_std_5) else 5.0) * 3.0)
                predictions.append({
                    'driver_id': driver_id,
                    'constructor_id': constructor_id,
                    'driver_name': driver_name_map.get(driver_id, f"Driver {driver_id}"),
                    'team_name': constructor_name_map.get(constructor_id, f"Team {constructor_id}"),
                    'predicted_position': ensemble_pred,
                    'confidence': round(confidence, 1),
                    'model_predictions': model_predictions,
                    'driver_stats': {
                        'avg_position': round(driver_avg_5, 2),
                        'recent_form': round(driver_recent_form, 2),
                        'total_points': season_points_so_far,
                        'circuit_avg': round(driver_circuit_avg, 2)
                    }
                })
            # Sort by predicted position
            predictions.sort(key=lambda x: x['predicted_position'])
            # Assign final positions
            for i, pred in enumerate(predictions):
                pred['final_position'] = i + 1
            return predictions
        except Exception as e:
            print(f"Error in real data prediction: {e}")
            return self._fallback_prediction(race_conditions)
    
    def _fallback_prediction(self, race_conditions):
        """Fallback prediction when real data is not available"""
        print("Using fallback prediction...")
        
        # Create a simple prediction based on race conditions
        base_position = 10
        if 'monaco' in race_conditions.get('race', '').lower():
            base_position = 8
        elif 'spa' in race_conditions.get('race', '').lower():
            base_position = 12
        
        return [{
            'driver_id': 1,
            'driver_name': 'Max Verstappen',
            'team_name': 'Red Bull',
            'predicted_position': base_position,
            'final_position': 1,
            'confidence': 75.0,
            'model_predictions': {'random_forest': base_position, 'xgboost': base_position, 'neural_network': base_position},
            'driver_stats': {'avg_position': 8.5, 'recent_form': 7.2, 'total_points': 100, 'circuit_avg': 8.0}
        }]

    def _get_driver_team_stats(self, race_conditions):
        """Get real driver and team stats for the race"""
        try:
            results = pd.read_csv('data/results.csv')
            races = pd.read_csv('data/races.csv')
            
            # Find the race
            race_row = races[races['name'].str.lower() == race_conditions.get('race', '').lower()].iloc[0] if not races.empty and 'name' in races.columns and race_conditions.get('race') else None
            
            if race_row is not None and 'raceId' in race_row:
                race_id = race_row['raceId']
                year = race_row['year']
                
                # Get results for this race
                race_results = results[results['raceId'] == race_id]
                
                if not race_results.empty:
                    # Calculate driver stats (using recent races before this one)
                    recent_races = races[(races['year'] == year) & (races['round'] < race_row['round'])]
                    recent_race_ids = recent_races['raceId'].tolist()
                    
                    if recent_race_ids:
                        recent_results = results[results['raceId'].isin(recent_race_ids)]
                        
                        # Calculate driver averages
                        recent_results['position'] = pd.to_numeric(recent_results['position'], errors='coerce')
                        recent_results['points'] = pd.to_numeric(recent_results['points'], errors='coerce')
                        
                        driver_stats = recent_results.groupby('driverId').agg({
                            'position': ['mean', 'std'],
                            'points': 'sum'
                        }).reset_index()
                        
                        driver_stats.columns = ['driverId', 'avg_position', 'position_std', 'total_points']
                        
                        # Calculate team stats
                        team_stats = recent_results.groupby('constructorId').agg({
                            'position': 'mean',
                            'points': 'mean'
                        }).reset_index()
                        
                        team_stats.columns = ['constructorId', 'team_avg_position', 'team_avg_points']
                        
                        return {
                            'driver_stats': driver_stats,
                            'team_stats': team_stats,
                            'race_results': race_results
                        }
            
        except Exception as e:
            print(f"Warning: Could not load driver/team stats: {e}")
        
        return None

    def _create_race_features(self, race_conditions):
        """Create feature vector for race prediction using real race/circuit data"""
        # Load races and circuits data
        try:
            races = pd.read_csv('data/races.csv')
            circuits = pd.read_csv('data/circuits.csv')
        except Exception as e:
            print(f"Warning: Could not load races/circuits data: {e}")
            races = pd.DataFrame()
            circuits = pd.DataFrame()

        # Find the race row
        race_row = races[races['name'].str.lower() == race_conditions.get('race', '').lower()].iloc[0] if not races.empty and 'name' in races.columns and race_conditions.get('race') else None
        # Find the circuit row
        circuit_row = circuits[circuits['circuitId'] == race_row['circuitId']].iloc[0] if race_row is not None and 'circuitId' in race_row and not circuits.empty else None

        # Get driver/team stats
        stats = self._get_driver_team_stats(race_conditions)
        
        # Use real values if available, else fallback
        grid = race_conditions.get('grid', 10.0)
        year = race_row['year'] if race_row is not None and 'year' in race_row else race_conditions.get('year', 2025)
        round_num = race_row['round'] if race_row is not None and 'round' in race_row else 1
        lat = circuit_row['lat'] if circuit_row is not None and 'lat' in circuit_row else 26.0
        lng = circuit_row['lng'] if circuit_row is not None and 'lng' in circuit_row else 50.0
        alt = circuit_row['alt'] if circuit_row is not None and 'alt' in circuit_row else 0.0

        # Use real driver/team stats if available
        driver_avg_5races = 8.5
        driver_avg_10races = 9.0
        driver_recent_form = 8.0
        driver_circuit_avg = 8.0
        team_avg_position = 7.5
        team_avg_points = 15.0
        
        if stats and not stats['driver_stats'].empty:
            # Use average of all drivers for this prediction
            driver_avg_5races = stats['driver_stats']['avg_position'].mean()
            driver_avg_10races = stats['driver_stats']['avg_position'].mean()
            driver_recent_form = stats['driver_stats']['avg_position'].mean()
            driver_circuit_avg = stats['driver_stats']['avg_position'].mean()
            
        if stats and not stats['team_stats'].empty:
            team_avg_position = stats['team_stats']['team_avg_position'].mean()
            team_avg_points = stats['team_stats']['team_avg_points'].mean()

        features = [
            grid,
            (year - 1950) / 100,
            round_num / 25,
            lat,
            lng,
            alt,
            driver_avg_5races,
            driver_avg_10races,
            driver_recent_form,
            driver_circuit_avg,
            team_avg_position,
            team_avg_points,
            7.0,   # circuit_difficulty
            1.0,   # season_race_number
            0.0,   # season_points_so_far
            -2.0,  # grid_advantage
            1.0,   # driverId_encoded
            1.0,   # constructorId_encoded
            1.0    # circuitId_encoded
        ]
        # Ensure correct number of features
        if len(features) < len(self.feature_names):
            features.extend([0.0] * (len(self.feature_names) - len(features)))
        elif len(features) > len(self.feature_names):
            features = features[:len(self.feature_names)]
        
        # Ensure we have at least 19 features (the expected number)
        while len(features) < 19:
            features.append(0.0)
        
        return features
    
    def train_all(self):
        """Complete training pipeline"""
        print("🚀 Starting robust F1 prediction training...")
        
        # Load and clean data
        df_train, df_predict = self.load_and_clean_data()
        
        # Engineer features
        df_train = self.engineer_robust_features(df_train)
        df_predict = self.engineer_robust_features(df_predict)
        
        # Encode categorical features
        df_train = self.encode_categorical_features(df_train)
        df_predict = self.encode_categorical_features(df_predict)
        
        # Prepare features
        X_train, y_train = self.prepare_features(df_train)
        X_predict, y_predict = self.prepare_features(df_predict)
        
        # Train models
        results = self.train_robust_models(X_train, y_train)
        
        print("✅ Robust F1 prediction training completed!")
        return results

def main():
    """Main function"""
    predictor = RobustF1Predictor()
    
    # Train models
    results = predictor.train_all()
    
    # Test prediction
    race_conditions = {
        'race': 'Bahrain Grand Prix',
        'year': 2025,
        'temperature': 25,
        'humidity': 60,
        'rain_probability': 0.1
    }
    
    prediction = predictor.predict_race_with_real_data(race_conditions)
    
    print(f"\n🏆 Test prediction for {race_conditions['race']}:")
    for p in prediction:
        print(f"   Driver: {p['driver_name']} (Team: {p['team_name']})")
        print(f"   Predicted Position: {p['predicted_position']}")
        print(f"   Confidence: {p['confidence']}%")
        print(f"   Model Agreement: {p['model_predictions']}")
    
    return predictor, results, prediction

if __name__ == "__main__":
    main() 