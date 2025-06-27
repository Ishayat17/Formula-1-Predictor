import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import json
from datetime import datetime, timedelta
from advanced_f1_data_loader import AdvancedF1DataLoader

class WorldClassF1Preprocessor:
    def __init__(self):
        self.loader = AdvancedF1DataLoader()
        self.master_df = None
        self.encoders = {}
        self.scalers = {}
        self.feature_selector = None
        
    def load_comprehensive_data(self):
        """Load all F1 data sources"""
        print("Loading comprehensive F1 data...")
        all_data = self.loader.load_all_data()
        self.master_df = all_data['master_dataset']
        self.upcoming_races = all_data['upcoming_races']
        self.current_standings = all_data['current_standings']
        
        if self.master_df is None:
            raise ValueError("Failed to create master dataset")
            
        print(f"Master dataset loaded: {len(self.master_df)} records, {len(self.master_df.columns)} columns")
        return all_data
    
    def advanced_data_cleaning(self, df):
        """Advanced data cleaning with F1 domain knowledge"""
        print("Performing advanced data cleaning...")
        
        # Convert data types with F1-specific handling
        numeric_columns = [
            'grid', 'laps', 'points', 'year', 'round', 'lat', 'lng', 'alt',
            'quali_position', 'championship_position', 'championship_points',
            'constructor_championship_position', 'constructor_championship_points',
            'pit_stops_count', 'avg_pit_duration', 'total_pit_ms'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle F1-specific position data
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['finished_race'] = df['position_numeric'].notna().astype(int)
        df['points_scored'] = (df['points'] > 0).astype(int)
        
        # Parse lap times with F1 format handling
        def parse_f1_time(time_str):
            if pd.isna(time_str) or time_str == '\\N' or time_str == '':
                return np.nan
            try:
                time_str = str(time_str).strip()
                if ':' in time_str:
                    # Handle MM:SS.mmm format
                    parts = time_str.split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                elif time_str.startswith('+'):
                    # Handle +SS.mmm format (gap to leader)
                    return float(time_str[1:])
                else:
                    return float(time_str)
            except:
                return np.nan
        
        # Parse qualifying times
        for q_col in ['q1', 'q2', 'q3']:
            if q_col in df.columns:
                df[f'{q_col}_seconds'] = df[q_col].apply(parse_f1_time)
        
        # Parse fastest lap times
        if 'fastestLapTime' in df.columns:
            df['fastest_lap_seconds'] = df['fastestLapTime'].apply(parse_f1_time)
        
        # Handle DNF and status information
        if 'statusId' in df.columns:
            # Status 1 = Finished, others are various DNF reasons
            df['dnf'] = (df['statusId'] != '1').astype(int)
            df['mechanical_failure'] = df['statusId'].isin(['2', '3', '4', '5', '6']).astype(int)
            df['accident'] = df['statusId'].isin(['20', '21', '22', '23']).astype(int)
        
        # Clean pit stop data
        pit_columns = ['pit_stops_count', 'avg_pit_duration', 'total_pit_ms']
        for col in pit_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)  # No pit stops = 0
        
        # Handle missing championship data (early career drivers)
        championship_cols = ['championship_position', 'championship_points']
        for col in championship_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Remove obviously invalid data
        df = df[df['year'] >= 1950]  # F1 started in 1950
        df = df[df['round'] <= 25]   # Max ~25 races per season
        
        print(f"Data cleaning completed. Dataset shape: {df.shape}")
        return df
    
    def create_world_class_features(self, df):
        """Create world-class F1 prediction features"""
        print("Creating world-class F1 features...")
        
        # Sort data for time-series features
        df_sorted = df.sort_values(['driverId', 'year', 'round'])
        
        # === DRIVER PERFORMANCE FEATURES ===
        
        # Historical performance metrics (rolling windows)
        for window in [3, 5, 10]:
            df[f'driver_avg_position_{window}'] = df_sorted.groupby('driverId')['position_numeric'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'driver_avg_points_{window}'] = df_sorted.groupby('driverId')['points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'driver_finish_rate_{window}'] = df_sorted.groupby('driverId')['finished_race'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Driver consistency metrics
        df['driver_position_std'] = df_sorted.groupby('driverId')['position_numeric'].transform(
            lambda x: x.rolling(window=10, min_periods=3).std().shift(1)
        )
        
        # Driver momentum (recent trend)
        df['driver_momentum'] = df_sorted.groupby('driverId')['position_numeric'].transform(
            lambda x: x.rolling(window=3, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            ).shift(1)
        )
        
        # Career stage features
        df['career_race_count'] = df_sorted.groupby('driverId').cumcount()
        df['driver_experience'] = df['career_race_count'] / 100  # Normalize
        
        # === CONSTRUCTOR PERFORMANCE FEATURES ===
        
        # Constructor rolling performance
        for window in [3, 5, 10]:
            df[f'constructor_avg_position_{window}'] = df_sorted.groupby('constructorId')['position_numeric'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'constructor_avg_points_{window}'] = df_sorted.groupby('constructorId')['points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Constructor reliability
        df['constructor_dnf_rate'] = df_sorted.groupby('constructorId')['dnf'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
        )
        
        # === CIRCUIT-SPECIFIC FEATURES ===
        
        # Driver performance at specific circuits
        df['driver_circuit_avg'] = df_sorted.groupby(['driverId', 'circuitId'])['position_numeric'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['driver_circuit_best'] = df_sorted.groupby(['driverId', 'circuitId'])['position_numeric'].transform(
            lambda x: x.expanding().min().shift(1)
        )
        
        # Constructor performance at circuits
        df['constructor_circuit_avg'] = df_sorted.groupby(['constructorId', 'circuitId'])['position_numeric'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Circuit characteristics impact
        if 'alt' in df.columns:
            df['high_altitude'] = (df['alt'] > 1000).astype(int)
        if 'lat' in df.columns:
            df['tropical_circuit'] = (abs(df['lat']) < 23.5).astype(int)
        
        # === QUALIFYING PERFORMANCE FEATURES ===
        
        # Qualifying vs race performance
        df['quali_race_diff'] = df['position_numeric'] - df['quali_position']
        df['grid_position_diff'] = df['position_numeric'] - df['grid']
        
        # Qualifying progression (Q1 -> Q2 -> Q3)
        df['quali_progression'] = 0
        if 'q1_seconds' in df.columns and 'q2_seconds' in df.columns:
            df['q1_q2_improvement'] = df['q1_seconds'] - df['q2_seconds']
        if 'q2_seconds' in df.columns and 'q3_seconds' in df.columns:
            df['q2_q3_improvement'] = df['q2_seconds'] - df['q3_seconds']
        
        # Best qualifying time relative to pole
        df['quali_gap_to_pole'] = df.groupby('raceId')['q1_seconds'].transform(
            lambda x: x - x.min() if x.notna().any() else np.nan
        )
        
        # === PIT STOP STRATEGY FEATURES ===
        
        if 'pit_stops_count' in df.columns:
            # Pit stop efficiency
            df['pit_stop_efficiency'] = df['avg_pit_duration'] / (df['pit_stops_count'] + 1)
            
            # Pit strategy relative to field
            df['pit_stops_vs_avg'] = df.groupby('raceId')['pit_stops_count'].transform(
                lambda x: x - x.mean()
            )
            
            # Pit window optimization
            df['optimal_pit_count'] = df.groupby(['circuitId', 'year'])['pit_stops_count'].transform(
                lambda x: x.mode().iloc[0] if not x.empty else 2
            )
            df['pit_strategy_deviation'] = abs(df['pit_stops_count'] - df['optimal_pit_count'])
        
        # === CHAMPIONSHIP CONTEXT FEATURES ===
        
        # Championship battle intensity
        if 'championship_position' in df.columns:
            df['championship_battle'] = (df['championship_position'] <= 3).astype(int)
            df['title_contender'] = (df['championship_position'] <= 2).astype(int)
            
            # Points gap to leader
            df['points_gap_to_leader'] = df.groupby('raceId')['championship_points'].transform(
                lambda x: x.max() - x
            )
            
            # Championship pressure (races remaining vs points gap)
            df['championship_pressure'] = df['points_gap_to_leader'] / (25 - df['round'] + 1)
        
        # === SEASON PROGRESSION FEATURES ===
        
        # Season stage
        df['season_stage'] = pd.cut(df['round'], bins=[0, 5, 15, 25], labels=['early', 'mid', 'late'])
        df['season_stage_encoded'] = df['season_stage'].cat.codes
        
        # Season momentum
        df['season_points_so_far'] = df_sorted.groupby(['driverId', 'year'])['points'].cumsum().shift(1)
        df['season_avg_position'] = df_sorted.groupby(['driverId', 'year'])['position_numeric'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # === WEATHER AND CONDITIONS ===
        
        # Simulate weather impact (in production, use real weather data)
        np.random.seed(42)
        df['weather_risk'] = np.random.uniform(0, 1, len(df))
        df['rain_probability'] = np.random.uniform(0, 0.3, len(df))
        
        # === TEAM DYNAMICS ===
        
        # Teammate comparison
        teammate_stats = df.groupby(['constructorId', 'raceId']).agg({
            'position_numeric': ['mean', 'std'],
            'points': 'sum'
        }).reset_index()
        teammate_stats.columns = ['constructorId', 'raceId', 'teammate_avg_pos', 'teammate_pos_diff', 'team_race_points']
        df = df.merge(teammate_stats, on=['constructorId', 'raceId'], how='left')
        
        # === ADVANCED STATISTICAL FEATURES ===
        
        # ELO-style rating system
        df['driver_elo'] = 1500  # Starting ELO
        df['constructor_elo'] = 1500
        
        # Momentum indicators
        df['hot_streak'] = df_sorted.groupby('driverId')['points_scored'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
        )
        
        # Clutch performance (performance in high-pressure situations)
        df['clutch_performance'] = df['position_numeric'] * df['championship_pressure']
        
        print(f"Feature engineering completed. Dataset shape: {df.shape}")
        print(f"Total features created: {len(df.columns)}")
        
        return df
    
    def encode_categorical_features(self, df):
        """Advanced categorical encoding"""
        print("Encoding categorical features...")
        
        # High cardinality categorical features
        high_card_features = ['driverId', 'constructorId', 'circuitId']
        
        # Medium cardinality features
        medium_card_features = ['nationality', 'nationality_driver', 'country']
        
        # Low cardinality features
        low_card_features = ['season_stage_encoded']
        
        # Target encoding for high cardinality features
        for feature in high_card_features:
            if feature in df.columns:
                # Calculate target mean for each category
                target_mean = df.groupby(feature)['position_numeric'].mean()
                df[f'{feature}_target_encoded'] = df[feature].map(target_mean)
                
                # Add noise to prevent overfitting
                noise = np.random.normal(0, 0.1, len(df))
                df[f'{feature}_target_encoded'] += noise
                
                # Standard label encoding as backup
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.encoders[feature] = le
        
        # One-hot encoding for low cardinality features
        for feature in low_card_features:
            if feature in df.columns and df[feature].nunique() < 10:
                dummies = pd.get_dummies(df[feature], prefix=feature)
                df = pd.concat([df, dummies], axis=1)
        
        # Frequency encoding for medium cardinality features
        for feature in medium_card_features:
            if feature in df.columns:
                freq_map = df[feature].value_counts().to_dict()
                df[f'{feature}_frequency'] = df[feature].map(freq_map)
        
        return df
    
    def select_best_features(self, X, y, k=50):
        """Select the best features using statistical tests"""
        print(f"Selecting top {k} features...")
        
        # Remove non-numeric columns
        numeric_X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_X = numeric_X.fillna(numeric_X.median())
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(k, numeric_X.shape[1]))
        X_selected = selector.fit_transform(numeric_X, y)
        
        # Get selected feature names
        selected_features = numeric_X.columns[selector.get_support()].tolist()
        
        self.feature_selector = selector
        
        print(f"Selected {len(selected_features)} features")
        print("Top 10 features:", selected_features[:10])
        
        return X_selected, selected_features
    
    def normalize_features(self, X_train, X_test):
        """Advanced feature normalization"""
        print("Normalizing features with robust scaling...")
        
        # Use RobustScaler to handle outliers better
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['robust'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def prepare_prediction_features(self, df):
        """Prepare final feature set for world-class predictions"""
        print("Preparing world-class prediction features...")
        
        # Define comprehensive feature set
        prediction_features = [
            # Core performance features
            'grid', 'quali_position', 'year', 'round',
            
            # Driver performance features
            'driver_avg_position_3', 'driver_avg_position_5', 'driver_avg_position_10',
            'driver_avg_points_3', 'driver_avg_points_5', 'driver_avg_points_10',
            'driver_finish_rate_3', 'driver_finish_rate_5', 'driver_finish_rate_10',
            'driver_position_std', 'driver_momentum', 'driver_experience',
            
            # Constructor features
            'constructor_avg_position_3', 'constructor_avg_position_5', 'constructor_avg_position_10',
            'constructor_avg_points_3', 'constructor_avg_points_5', 'constructor_avg_points_10',
            'constructor_dnf_rate',
            
            # Circuit-specific features
            'driver_circuit_avg', 'driver_circuit_best', 'constructor_circuit_avg',
            'lat', 'lng', 'alt', 'high_altitude', 'tropical_circuit',
            
            # Qualifying features
            'quali_race_diff', 'grid_position_diff', 'quali_gap_to_pole',
            
            # Pit stop features
            'pit_stops_count', 'pit_stop_efficiency', 'pit_stops_vs_avg', 'pit_strategy_deviation',
            
            # Championship context
            'championship_position', 'championship_points', 'championship_battle',
            'title_contender', 'points_gap_to_leader', 'championship_pressure',
            
            # Season features
            'season_stage_encoded', 'season_points_so_far', 'season_avg_position',
            
            # Weather and conditions
            'weather_risk', 'rain_probability',
            
            # Team dynamics
            'teammate_avg_pos', 'teammate_pos_diff', 'team_race_points',
            
            # Advanced features
            'driver_elo', 'constructor_elo', 'hot_streak', 'clutch_performance',
            
            # Encoded categorical features
            'driverId_target_encoded', 'constructorId_target_encoded', 'circuitId_target_encoded',
            'driverId_encoded', 'constructorId_encoded', 'circuitId_encoded'
        ]
        
        # Filter available features
        available_features = [col for col in prediction_features if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle missing values with advanced imputation
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
        
        # Target variable
        y = df['position_numeric'].fillna(21)  # DNF = 21st position
        
        # Remove rows where target is missing
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features used: {len(available_features)}")
        
        return X, y, available_features
    
    def process_world_class_data(self):
        """Main world-class data processing pipeline"""
        print("Starting world-class F1 data processing pipeline...")
        
        # Load comprehensive data
        all_data = self.load_comprehensive_data()
        
        # Advanced data cleaning
        df = self.advanced_data_cleaning(self.master_df)
        
        # Create world-class features
        df = self.create_world_class_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare prediction features
        X, y, feature_names = self.prepare_prediction_features(df)
        
        # Split data with stratification by year to prevent data leakage
        # Use recent years for testing to simulate real-world prediction
        recent_years = df['year'] >= 2020
        X_recent = X[recent_years]
        y_recent = y[recent_years]
        X_historical = X[~recent_years]
        y_historical = y[~recent_years]
        
        # Split historical data for training/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_historical, y_historical, test_size=0.2, random_state=42
        )
        
        # Use recent data as final test set
        X_test, y_test = X_recent, y_recent
        
        # Feature selection
        X_train_selected, selected_features = self.select_best_features(X_train, y_train, k=50)
        X_val_selected = self.feature_selector.transform(X_val[selected_features])
        X_test_selected = self.feature_selector.transform(X_test[selected_features])
        
        # Normalize features
        X_train_final, X_val_final = self.normalize_features(X_train_selected, X_val_selected)
        X_test_final = self.scalers['robust'].transform(X_test_selected)
        
        # Save processed data
        np.save('X_train_world_class.npy', X_train_final)
        np.save('X_val_world_class.npy', X_val_final)
        np.save('X_test_world_class.npy', X_test_final)
        np.save('y_train_world_class.npy', y_train)
        np.save('y_val_world_class.npy', y_val)
        np.save('y_test_world_class.npy', y_test)
        
        # Save metadata
        metadata = {
            'feature_names': selected_features,
            'total_features': len(selected_features),
            'training_samples': len(X_train_final),
            'validation_samples': len(X_val_final),
            'test_samples': len(X_test_final),
            'data_split_strategy': 'temporal',
            'feature_selection_method': 'SelectKBest',
            'scaling_method': 'RobustScaler'
        }
        
        with open('world_class_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save encoders and scalers
        joblib.dump(self.encoders, 'world_class_encoders.pkl')
        joblib.dump(self.scalers, 'world_class_scalers.pkl')
        joblib.dump(self.feature_selector, 'world_class_feature_selector.pkl')
        
        # Save processed dataframe for analysis
        df.to_csv('world_class_f1_data.csv', index=False)
        
        print("World-class F1 data processing completed!")
        print(f"Training set: {len(X_train_final)} samples")
        print(f"Validation set: {len(X_val_final)} samples")
        print(f"Test set: {len(X_test_final)} samples")
        print(f"Selected features: {len(selected_features)}")
        
        return {
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': selected_features,
            'metadata': metadata,
            'upcoming_races': self.upcoming_races,
            'current_standings': self.current_standings
        }

def main():
    """Main function"""
    preprocessor = WorldClassF1Preprocessor()
    processed_data = preprocessor.process_world_class_data()
    
    print("\nWorld-class F1 data preprocessing completed!")
    print("Ready for world-class model training!")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()
