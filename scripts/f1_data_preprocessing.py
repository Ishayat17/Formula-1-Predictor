import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
from f1_data_loader import F1DataLoader

class F1DataPreprocessor:
    def __init__(self):
        self.loader = F1DataLoader()
        self.dataframes = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_merge_data(self):
        """Load and merge F1 datasets"""
        print("Loading F1 data...")
        self.dataframes = self.loader.load_data()
        
        # Main results dataframe
        results = self.dataframes['results'].copy()
        races = self.dataframes['races'].copy()
        qualifying = self.dataframes['qualifying'].copy()
        circuits = self.dataframes['circuits'].copy()
        
        print("Merging datasets...")
        
        # Merge results with race information
        merged_data = results.merge(races, on='raceId', how='left')
        
        # Merge with qualifying data
        qualifying_agg = qualifying.groupby(['raceId', 'driverId']).agg({
            'position': 'first',
            'q1': 'first',
            'q2': 'first', 
            'q3': 'first'
        }).reset_index()
        qualifying_agg.rename(columns={'position': 'quali_position'}, inplace=True)
        
        merged_data = merged_data.merge(qualifying_agg, on=['raceId', 'driverId'], how='left')
        
        # Merge with circuit information
        merged_data = merged_data.merge(circuits, on='circuitId', how='left')
        
        # Add pit stop data
        if 'pit_stops' in self.dataframes:
            pit_stops = self.dataframes['pit_stops'].copy()
            pit_stop_stats = pit_stops.groupby(['raceId', 'driverId']).agg({
                'stop': 'count',
                'duration': 'mean',
                'milliseconds': 'mean'
            }).reset_index()
            pit_stop_stats.rename(columns={
                'stop': 'pit_stop_count',
                'duration': 'avg_pit_duration',
                'milliseconds': 'avg_pit_milliseconds'
            }, inplace=True)
            
            merged_data = merged_data.merge(pit_stop_stats, on=['raceId', 'driverId'], how='left')
        
        print(f"Merged dataset shape: {merged_data.shape}")
        return merged_data
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("Cleaning data...")
        
        # Convert data types
        numeric_columns = ['grid', 'laps', 'points', 'year', 'round', 'lat', 'lng', 'alt']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle position data
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['finished'] = df['position_numeric'].notna().astype(int)
        
        # Handle qualifying positions
        df['quali_position'] = pd.to_numeric(df['quali_position'], errors='coerce')
        
        # Parse lap times
        def parse_lap_time(time_str):
            if pd.isna(time_str) or time_str == '\\N':
                return np.nan
            try:
                if ':' in str(time_str):
                    parts = str(time_str).split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_str)
            except:
                return np.nan
        
        # Parse qualifying times
        for q_col in ['q1', 'q2', 'q3']:
            if q_col in df.columns:
                df[f'{q_col}_seconds'] = df[q_col].apply(parse_lap_time)
        
        # Handle missing values
        df['pit_stop_count'] = df['pit_stop_count'].fillna(0)
        df['avg_pit_duration'] = df['avg_pit_duration'].fillna(df['avg_pit_duration'].median())
        df['avg_pit_milliseconds'] = df['avg_pit_milliseconds'].fillna(df['avg_pit_milliseconds'].median())
        
        # Remove rows with critical missing data
        df = df.dropna(subset=['driverId', 'constructorId', 'raceId'])
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Create advanced features for F1 prediction"""
        print("Engineering features...")
        
        # Driver performance features
        df_sorted = df.sort_values(['driverId', 'year', 'round'])
        
        # Historical performance (rolling averages)
        df['driver_avg_position'] = df_sorted.groupby('driverId')['position_numeric'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        df['driver_avg_points'] = df_sorted.groupby('driverId')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        df['driver_finish_rate'] = df_sorted.groupby('driverId')['finished'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
        )
        
        # Constructor performance
        df['constructor_avg_position'] = df_sorted.groupby('constructorId')['position_numeric'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        df['constructor_avg_points'] = df_sorted.groupby('constructorId')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Circuit-specific features
        df['driver_circuit_avg'] = df_sorted.groupby(['driverId', 'circuitId'])['position_numeric'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Grid vs finish performance
        df['grid_position_diff'] = df['position_numeric'] - df['grid']
        df['quali_race_diff'] = df['position_numeric'] - df['quali_position']
        
        # Season performance
        df['season_race_number'] = df.groupby(['driverId', 'year']).cumcount() + 1
        df['season_points_so_far'] = df_sorted.groupby(['driverId', 'year'])['points'].cumsum().shift(1)
        
        # Circuit characteristics
        df['circuit_difficulty'] = df.groupby('circuitId')['grid_position_diff'].transform('std')
        
        # Recent form (last 3 races)
        df['recent_avg_position'] = df_sorted.groupby('driverId')['position_numeric'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Qualifying performance
        df['best_quali_time'] = df[['q1_seconds', 'q2_seconds', 'q3_seconds']].min(axis=1)
        df['quali_performance'] = df.groupby('raceId')['best_quali_time'].rank()
        
        # Pit stop strategy features
        df['pit_stop_efficiency'] = df['avg_pit_duration'] / (df['pit_stop_count'] + 1)
        
        print(f"Feature engineering completed. Dataset shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        print("Encoding categorical features...")
        
        categorical_features = ['driverId', 'constructorId', 'circuitId', 'name', 'location', 'country']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.encoders[feature] = le
        
        return df
    
    def prepare_features(self, df):
        """Prepare final feature set for modeling"""
        print("Preparing features for modeling...")
        
        # Select features for modeling
        feature_columns = [
            'grid', 'quali_position', 'year', 'round', 'lat', 'lng', 'alt',
            'pit_stop_count', 'avg_pit_duration', 'pit_stop_efficiency',
            'driver_avg_position', 'driver_avg_points', 'driver_finish_rate',
            'constructor_avg_position', 'constructor_avg_points',
            'driver_circuit_avg', 'season_race_number', 'season_points_so_far',
            'circuit_difficulty', 'recent_avg_position', 'quali_performance',
            'driverId_encoded', 'constructorId_encoded', 'circuitId_encoded'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Target variable (position)
        y = df['position_numeric'].fillna(21)  # DNF = 21st position
        
        # Remove rows where target is missing
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features used: {list(X.columns)}")
        
        return X, y
    
    def normalize_features(self, X_train, X_test):
        """Normalize features"""
        print("Normalizing features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def process_data(self):
        """Main data processing pipeline"""
        print("Starting F1 data processing pipeline...")
        
        # Load and merge data
        df = self.load_and_merge_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        # Save processed data
        np.save('X_train_f1.npy', X_train_scaled)
        np.save('X_test_f1.npy', X_test_scaled)
        np.save('y_train_f1.npy', y_train)
        np.save('y_test_f1.npy', y_test)
        
        # Save feature names
        feature_names = list(X.columns)
        with open('feature_names_f1.json', 'w') as f:
            json.dump(feature_names, f)
        
        # Save encoders and scaler
        joblib.dump(self.encoders, 'f1_encoders.pkl')
        joblib.dump(self.scaler, 'f1_scaler.pkl')
        
        # Save original dataframe for analysis
        df.to_csv('processed_f1_data.csv', index=False)
        
        print("F1 data processing completed successfully!")
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Features: {len(feature_names)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def main():
    """Main function"""
    preprocessor = F1DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.process_data()
    
    print("\nData preprocessing completed!")
    print(f"Ready for model training with {len(feature_names)} features")
    
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    main()
