from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.current_drivers_predictor import CurrentDriversPredictor
import pandas as pd

app = Flask(__name__)
CORS(app)
predictor = CurrentDriversPredictor()
predictor.load_improved_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    race = data.get('race')
    year = data.get('year')
    temperature = data.get('temperature', 25)
    humidity = data.get('humidity', 60)
    rain_probability = data.get('rain_probability', 0.2)
    if not race or not year:
        return jsonify({'success': False, 'error': 'Race and year are required'}), 400
    race_conditions = {
        'race': race,
        'year': year,
        'temperature': temperature,
        'humidity': humidity,
        'rain_probability': rain_probability
    }
    predictions = predictor.predict_current_drivers_race(race_conditions)
    return jsonify({'success': True, 'predictions': predictions})

@app.route('/predict-all', methods=['POST'])
def predict_all():
    data = request.json or {}
    year = data.get('year')
    races_df = pd.read_csv('data/races.csv')
    if not year:
        year = races_df['year'].max()
    races_this_year = races_df[races_df['year'] == year]
    results = []
    for _, row in races_this_year.iterrows():
        race_name = row['name']
        race_conditions = {
            'race': race_name,
            'year': year,
            'temperature': data.get('temperature', 25),
            'humidity': data.get('humidity', 60),
            'rain_probability': data.get('rain_probability', 0.2)
        }
        predictions = predictor.predict_current_drivers_race(race_conditions)
        results.append({'race': race_name, 'predictions': predictions})
    return jsonify({'success': True, 'results': results})

@app.route('/')
def home():
    return "F1 Predictor API is running! Use POST /predict for predictions. Use POST /predict-all for all races."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 