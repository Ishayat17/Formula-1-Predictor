import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Simulate trained ML models with realistic behavior
class MockMLModel:
    def __init__(self, model_type, mae, r2):
        self.model_type = model_type
        self.mae = mae
        self.r2 = r2
        self.feature_importance = self._generate_feature_importance()
    
    def _generate_feature_importance(self):
        # Realistic feature importance for F1 predictions
        return {
            'driver_skill': 0.25,
            'team_performance': 0.20,
            'grid_position': 0.15,
            'rain_probability': 0.12,
            'recent_form': 0.10,
            'driver_experience': 0.08,
            'temperature': 0.05,
            'circuit_difficulty': 0.03,
            'championship_points': 0.02
        }
    
    def predict(self, X):
        """Simulate ML model prediction"""
        predictions = []
        
        for features in X:
            # Extract key features (assuming specific order)
            driver_skill = features[9] if len(features) > 9 else 0
            team_performance = features[10] if len(features) > 10 else 0
            grid_position = features[14] if len(features) > 14 else 10
            rain_probability = features[6] if len(features) > 6 else 0
            recent_form = features[13] if len(features) > 13 else 10
            driver_experience = features[11] if len(features) > 11 else 5
            temperature = features[4] if len(features) > 4 else 0
            
            # ML-based prediction calculation
            base_prediction = 10.5  # Average position
            
            # Driver skill impact (normalized from -2 to +2 range)
            base_prediction -= (driver_skill * 4)
            
            # Team performance impact
            base_prediction -= (team_performance * 3)
            
            # Grid position impact (normalized)
            base_prediction += (grid_position * 0.3)
            
            # Weather impact
            if rain_probability > 0:
                # Rain affects prediction based on driver skill
                rain_impact = rain_probability * (driver_skill - 0.5) * 5
                base_prediction -= rain_impact
            
            # Temperature impact
            temp_impact = abs(temperature) * 0.1
            base_prediction += temp_impact
            
            # Recent form impact
            base_prediction += (recent_form - 10) * 0.2
            
            # Experience impact
            base_prediction -= (driver_experience - 5) * 0.1
            
            # Add model-specific noise
            if self.model_type == 'random_forest':
                noise = np.random.normal(0, 1.5)
            elif self.model_type == 'xgboost':
                noise = np.random.normal(0, 1.2)
            else:  # gradient_boosting
                noise = np.random.normal(0, 1.8)
            
            base_prediction += noise
            
            # Ensure prediction is within valid range
            prediction = max(1, min(20, base_prediction))
            predictions.append(prediction)
        
        return np.array(predictions)

class F1MLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metadata = {}
        self.is_loaded = False
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize mock ML models with realistic performance"""
        print("Initializing F1 ML models...")
        
        # Create mock models with different characteristics
        self.models = {
            'random_forest': MockMLModel('random_forest', 2.34, 0.78),
            'xgboost': MockMLModel('xgboost', 2.18, 0.82),
            'gradient_boosting': MockMLModel('gradient_boosting', 2.45, 0.76)
        }
        
        # Mock encoders for categorical variables
        self.encoders = {
            'driver': {'Max Verstappen': 0, 'Lewis Hamilton': 1, 'Charles Leclerc': 2, 'Lando Norris': 3},
            'team': {'Red Bull': 0, 'Ferrari': 1, 'McLaren': 2, 'Mercedes': 3},
            'circuit': {'bahrain': 0, 'jeddah': 1, 'albert_park': 2, 'suzuka': 3}
        }
        
        # Mock scaler (identity transformation for simplicity)
        self.scalers = {'feature_scaler': 'mock_scaler'}
        
        # Mock metadata
        self.metadata = {
            'feature_names': [
                'driver_encoded', 'team_encoded', 'circuit_encoded', 'year',
                'temperature', 'humidity', 'rain_probability', 'circuit_difficulty',
                'overtaking_difficulty', 'driver_skill', 'team_performance',
                'driver_experience', 'championship_points', 'recent_form', 'grid_position'
            ],
            'best_model': 'xgboost',
            'metrics': {
                'random_forest': {'mae': 2.34, 'r2': 0.78},
                'xgboost': {'mae': 2.18, 'r2': 0.82},
                'gradient_boosting': {'mae': 2.45, 'r2': 0.76}
            }
        }
        
        self.is_loaded = True
        print("ML models initialized successfully!")
        
    def prepare_prediction_features(self, race_data):
        """Prepare features for ML prediction"""
        # Encode categorical features
        driver_encoded = self.encoders['driver'].get(race_data['driver'], 0)
        team_encoded = self.encoders['team'].get(race_data['team'], 0)
        circuit_encoded = self.encoders['circuit'].get(race_data['circuit'], 0)
        
        # Create feature vector matching training format
        features = [
            driver_encoded,
            team_encoded,
            circuit_encoded,
            race_data.get('year', 2025),
            race_data.get('temperature', 25) / 50,  # Normalize temperature
            race_data.get('humidity', 60) / 100,    # Normalize humidity
            race_data.get('rain_probability', 0.2),
            race_data.get('circuit_difficulty', 7) / 10,
            race_data.get('overtaking_difficulty', 6) / 10,
            race_data.get('driver_skill', 75) / 100,
            race_data.get('team_performance', 75) / 100,
            race_data.get('driver_experience', 5) / 25,
            race_data.get('championship_points', 100) / 500,
            race_data.get('recent_form', 10) / 20,
            race_data.get('grid_position', 10) / 20
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_position(self, race_data):
        """Predict race position using ML models"""
        # Prepare features
        X = self.prepare_prediction_features(race_data)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            predictions[name] = max(1, min(20, round(pred)))
        
        # Ensemble prediction (weighted average)
        weights = {'random_forest': 0.3, 'xgboost': 0.4, 'gradient_boosting': 0.3}
        ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)
        ensemble_pred = max(1, min(20, round(ensemble_pred)))
        
        # Calculate confidence based on model agreement
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        confidence = max(60, 95 - pred_std * 8)
        
        return {
            'predicted_position': ensemble_pred,
            'confidence': round(confidence, 1),
            'individual_predictions': predictions,
            'model_used': 'ensemble',
            'prediction_std': round(pred_std, 2)
        }
    
    def predict_race_results(self, drivers_data, race_conditions):
        """Predict full race results using ML"""
        print(f"🤖 Running ML predictions for {len(drivers_data)} drivers...")
        
        predictions = []
        
        for driver_data in drivers_data:
            # Combine driver data with race conditions
            race_data = {**driver_data, **race_conditions}
            
            # Get ML prediction
            ml_result = self.predict_position(race_data)
            
            predictions.append({
                'driver': driver_data['driver'],
                'team': driver_data['team'],
                'number': driver_data.get('number', 0),
                'driverId': driver_data.get('driverId', ''),
                'predicted_position': ml_result['predicted_position'],
                'confidence': ml_result['confidence'],
                'ml_models_used': list(ml_result['individual_predictions'].keys()),
                'model_agreement': ml_result['prediction_std'] < 1.5,
                'individual_predictions': ml_result['individual_predictions'],
                'prediction_uncertainty': ml_result['prediction_std']
            })
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        # Assign final positions and probabilities
        final_predictions = []
        for i, pred in enumerate(predictions):
            position = i + 1
            
            # Calculate probability based on ML confidence and position
            base_probability = max(5, 100 - i * 4)
            confidence_factor = pred['confidence'] / 100
            probability = base_probability * confidence_factor
            
            final_predictions.append({
                **pred,
                'position': position,
                'probability': round(probability, 1),
                'expectedPosition': pred['predicted_position'],
                'recentForm': [8, 9, 10],  # Mock recent form
                'reliability': 85,
                'championshipPoints': 100,
                'overallRating': 80,
                'experience': 5,
                'estimatedGridPosition': position + np.random.randint(-3, 4),
                'weatherImpact': 'Medium' if race_conditions.get('rain_probability', 0) > 0.3 else 'Low',
                'mlScore': round(100 - (position - 1) * 4.5, 1),
                'mlUncertainty': pred['prediction_uncertainty']
            })
        
        print(f"✅ ML predictions complete!")
        print(f"🥇 Predicted winner: {final_predictions[0]['driver']} ({final_predictions[0]['team']})")
        print(f"🥈 P2: {final_predictions[1]['driver']} ({final_predictions[1]['team']})")
        print(f"🥉 P3: {final_predictions[2]['driver']} ({final_predictions[2]['team']})")
        
        return final_predictions

# Global predictor instance
ml_predictor = F1MLPredictor()

def get_ml_predictions(drivers_data, race_conditions):
    """Get ML-based predictions"""
    return ml_predictor.predict_race_results(drivers_data, race_conditions)
