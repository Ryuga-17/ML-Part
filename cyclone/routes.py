import numpy as np
import pandas as pd
import joblib
from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model

# === Load once globally ===
model = load_model("cyclone/cyclone_lstm_model.h5")
scaler_X = joblib.load("cyclone/scaler_X.pkl")
scaler_y = joblib.load("cyclome/scaler_y.pkl")

cyclone_ = Blueprint('cyclone_', __name__)

def preprocess_input(data):
    try:
        df = pd.DataFrame([{
            'ISO_TIME': pd.to_datetime(data['ISO_TIME'], errors='coerce'),
            'LAT': float(data['LAT']),
            'LON': float(data['LON']),
            'STORM_SPEED': float(data['STORM_SPEED']),
            'STORM_DIR': float(data['STORM_DIR'])
        }])

        df['hour'] = df['ISO_TIME'].dt.hour.fillna(0).astype(int)
        df['month'] = df['ISO_TIME'].dt.month.fillna(0).astype(int)

        df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
        df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
        df['lat_lon_interaction'] = df['LAT'] * df['LON']
        df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
        df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']

        features = [
            'LAT', 'LON', 'STORM_SPEED', 'hour', 'month',
            'lat_lon_interaction', 'speed_lat_interaction', 'speed_lon_interaction',
            'dir_sin', 'dir_cos'
        ]

        X = df[features].values
        X_scaled = scaler_X.transform(X)
        return X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    except Exception as e:
        raise ValueError(f"Error in preprocessing input: {str(e)}")

@cyclone_.route('/predict-path', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR']
        missing_fields = [field for field in required if field not in data]

        if missing_fields:
            return jsonify({"error": f"Missing field(s): {', '.join(missing_fields)}"}), 400

        X = preprocess_input(data)
        y_pred_scaled = model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        return jsonify({
            "Predicted_LAT": round(float(y_pred[0][0]), 4),
            "Predicted_LON": round(float(y_pred[0][1]), 4)
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
