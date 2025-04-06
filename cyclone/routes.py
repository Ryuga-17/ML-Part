from flask import Blueprint, request, jsonify, Flask
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from flask_cors import CORS  # Importing CORS


from flask import Blueprint, request, jsonify
# your existing cyclone model logic here...
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

cyclone_ = Blueprint('cyclone_path', __name__)


# Load model and scalers
custom_objects = {"mse": MeanSquaredError()}
model = load_model("cyclone/cyclone_lstm_model.h5", custom_objects=custom_objects)
scaler_X = joblib.load("cyclone/scaler_X.pkl")
scaler_y = joblib.load("cyclone/scaler_y.pkl")

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['hour'] = df['ISO_TIME'].dt.hour.fillna(0).astype(int)
    df['month'] = df['ISO_TIME'].dt.month.fillna(0).astype(int)
    df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
    df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
    df['lat_lon_interaction'] = df['LAT'] * df['LON']
    df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
    df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']
    features = ['LAT', 'LON', 'STORM_SPEED', 'hour', 'month',
                'lat_lon_interaction', 'speed_lat_interaction', 'speed_lon_interaction',
                'dir_sin', 'dir_cos']
    X = df[features].values
    X_scaled = scaler_X.transform(X)
    return X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

@cyclone_.route('/predict-path', methods=['POST'])
def predict():
    try:
        data = request.json
        required = ['ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR']
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        X = preprocess_input(data)
        y_pred_scaled = model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        return jsonify({
            "Predicted_LAT": float(y_pred[0][0]),
            "Predicted_LON": float(y_pred[0][1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
