import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

def load_f1_data():
    """Load F1 data from various sources"""
    print("Loading F1 data...")
    
    # Simulate loading race results data
    race_data = {
        'race_id': range(1, 101),
        'driver_name': np.random.choice(['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 
                                       'Lando Norris', 'Sergio Perez'], 100),
        'team': np.random.choice(['Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Aston Martin'], 100),
        'grid_position': np.random.randint(1, 21, 100),
        'finish_position': np.random.randint(1, 21, 100),
        'points': np.random.randint(0, 26, 100),
        'fastest_lap': np.random.uniform(70, 90, 100),
        'weather_temp': np.random.uniform(15, 35, 100),
        'weather_humidity': np.random.uniform(30, 80, 100),
        'circuit_length': np.random.uniform(3.0, 7.0, 100),
        'circuit_corners': np.random.randint(8, 25, 100)
    }
    
    df = pd.DataFrame(race_data)
    print(f"Loaded {len(df)} race records")
    return df

def clean_data(df):
    """Clean and validate the data"""
    print("Cleaning data...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Remove outliers using IQR method
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Data cleaned. {len(df)} records remaining")
    return df

def feature_engineering(df):
    """Create new features for better predictions"""
    print("Engineering features...")
    
    # Performance metrics
    df['grid_to_finish_diff'] = df['grid_position'] - df['finish_position']
    df['points_per_race'] = df['points'] / df['race_id']
    df['lap_time_normalized'] = df['fastest_lap'] / df['circuit_length']
    
    # Weather impact
    df['weather_score'] = (df['weather_temp'] * 0.3) + (df['weather_humidity'] * 0.7)
    
    # Circuit difficulty
    df['circuit_difficulty'] = (df['circuit_corners'] / df['circuit_length']) * 100
    
    # Team performance encoding
    team_performance = df.groupby('team')['points'].mean()
    df['team_avg_points'] = df['team'].map(team_performance)
    
    # Driver consistency
    driver_consistency = df.groupby('driver_name')['finish_position'].std()
    df['driver_consistency'] = df['driver_name'].map(driver_consistency)
    
    print(f"Created {len(df.columns)} features")
    return df

def encode_categorical_features(df):
    """Encode categorical variables"""
    print("Encoding categorical features...")
    
    # Label encoding for categorical variables
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    df['driver_encoded'] = le_driver.fit_transform(df['driver_name'])
    df['team_encoded'] = le_team.fit_transform(df['team'])
    
    # Save encoders for later use
    encoders = {
        'driver_encoder': le_driver.classes_.tolist(),
        'team_encoder': le_team.classes_.tolist()
    }
    
    with open('encoders.json', 'w') as f:
        json.dump(encoders, f)
    
    return df

def normalize_features(df):
    """Normalize numerical features"""
    print("Normalizing features...")
    
    # Select numerical features for scaling
    numerical_features = ['grid_position', 'fastest_lap', 'weather_temp', 'weather_humidity', 
                         'circuit_length', 'circuit_corners', 'grid_to_finish_diff',
                         'points_per_race', 'lap_time_normalized', 'weather_score',
                         'circuit_difficulty', 'team_avg_points', 'driver_consistency']
    
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("Features normalized")
    return df

def prepare_training_data(df):
    """Prepare data for machine learning"""
    print("Preparing training data...")
    
    # Define features and target
    feature_columns = ['grid_position', 'fastest_lap', 'weather_temp', 'weather_humidity',
                      'circuit_length', 'circuit_corners', 'driver_encoded', 'team_encoded',
                      'grid_to_finish_diff', 'points_per_race', 'lap_time_normalized',
                      'weather_score', 'circuit_difficulty', 'team_avg_points', 'driver_consistency']
    
    X = df[feature_columns]
    y = df['finish_position']  # Predicting finish position
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main preprocessing pipeline"""
    print("Starting F1 data preprocessing pipeline...")
    
    # Load data
    df = load_f1_data()
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Normalize features
    df = normalize_features(df)
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    
    # Save processed data
    df.to_csv('processed_f1_data.csv', index=False)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    print("Data preprocessing completed successfully!")
    print(f"Processed data saved with {len(df)} records and {len(df.columns)} features")

if __name__ == "__main__":
    main()
