from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.current_drivers_predictor import CurrentDriversPredictor

app = Flask(__name__)
CORS(app)
predictor = CurrentDriversPredictor()
predictor.load_improved_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    race = data.get('race', 'Monaco Grand Prix')
    year = data.get('year', 2024)
    temperature = data.get('temperature', 25)
    humidity = data.get('humidity', 60)
    rain_probability = data.get('rain_probability', 0.2)
    race_conditions = {
        'race': race,
        'year': year,
        'temperature': temperature,
        'humidity': humidity,
        'rain_probability': rain_probability
    }
    predictions = predictor.predict_current_drivers_race(race_conditions)
    return jsonify({'success': True, 'predictions': predictions})

@app.route('/')
def home():
    return "F1 Predictor API is running! Use POST /predict for predictions."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 