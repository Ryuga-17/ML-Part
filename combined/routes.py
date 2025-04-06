from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle
from flask import Blueprint

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Autoencoder Model Definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ====== Load Models Once at Startup ======
try:
    # Load models
    scaler = joblib.load("combined/scaler.pkl")
    custom_objects = {"mse": MeanSquaredError()}
    lstm_model = load_model("combined/cyclone_lstm_model.h5", custom_objects=custom_objects)
    speed_model = joblib.load("combined/speed_model.pkl")
    dir_model = joblib.load("combined/dir_model.pkl")
    scaler_X = joblib.load("combined/scaler_X.pkl")
    scaler_y = joblib.load("combined/scaler_y.pkl")
    encoder = load_model("combined/severity_encoder.h5")
    scaler_severity = joblib.load("combined/severity_scaler.pkl")
    kmeans = joblib.load("combined/severity_kmeans.pkl")
    kmeans_eq = joblib.load("combined/knn_model.pkl")

    

    # Load the Autoencoder model with weights_only set to True (safe globals added)
    model_path = 'combined/autoencoder2.pth'
    autoencoder = Autoencoder(input_dim=4, hidden_dim=16, latent_dim=64)
    lstm_model.compile(optimizer='adam', loss='mse')

    try:
        autoencoder.load_state_dict(torch.load(model_path))
        autoencoder.eval()  # Set model to evaluation mode
        logging.info("Autoencoder model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading Autoencoder model: {e}")
    
    severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}
    logging.info("All models loaded successfully!")

except Exception as e:
    logging.error(f"Error loading models: {e}")

# --------- Preprocessing ---------
def preprocess_input(data: dict, task='path'):
    df = pd.DataFrame([data])
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['HOUR'] = df['ISO_TIME'].dt.hour
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
    df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
    df['lat_lon_interaction'] = df['LAT'] * df['LON']
    df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
    df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']

    features = [
        'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos',
        'lat_lon_interaction', 'speed_lat_interaction', 'speed_lon_interaction'
    ]

    X = df[features].values

    if task == 'path':
        X = scaler_X.transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X

def preprocess_xgboost_input(data: dict):
    df = pd.DataFrame([data])
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['HOUR'] = df['ISO_TIME'].dt.hour
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
    df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))
    df['lat_lon_interaction'] = df['LAT'] * df['LON']
    df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
    df['STORM_SPEED_LAG1'] = df['STORM_SPEED']
    df['LAT_LAG'] = df['LAT']
    df['LON_LAG'] = df['LON']
    df['SPEED_MA3'] = df['STORM_SPEED']

    features = [
        'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos',
        'STORM_SPEED_LAG1', 'LAT_LAG', 'LON_LAG', 'SPEED_MA3',
        'lat_lon_interaction', 'speed_lat_interaction'
    ]

    X = df[features].values
    return X

# --------- Routes ---------
@app.route("/")
def home():
    return jsonify({"message": "Disaster ML API is running"})

predict_earthquake_bp = Blueprint('earthquake_predict', __name__)

@predict_earthquake_bp.route("/predict", methods=["POST"])
def predict_earthquake():
    try:
        data = request.get_json()
        values = np.array([[data['magnitude'], data['depth'], data['latitude'], data['longitude']]])
        input_scaled = scaler.transform(values)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            encoded_features = autoencoder.encoder(input_tensor).numpy()
            reconstructed_data = autoencoder(input_tensor)
        reconstruction_error = torch.mean((input_tensor - reconstructed_data) ** 2).item()
        threshold = 0.01
        anomaly = reconstruction_error > threshold
        cluster_label = kmeans_eq.predict(encoded_features)[0]
        severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}
        severity = severity_labels.get(cluster_label, "Unknown")

        return jsonify({"severity": severity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

speed_bp = Blueprint('speed', __name__)

@speed_bp.route("/predict-speed", methods=["POST"])
def predict_speed():
    try:
        data = request.get_json()
        X = preprocess_xgboost_input(data)
        speed_pred = speed_model.predict(X)
        dir_pred = dir_model.predict(X)
        return jsonify({
            "predicted_speed": speed_pred.tolist(),
            "predicted_direction": dir_pred.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

severity_bp = Blueprint('severity', __name__)

@severity_bp.route("/classify-severity", methods=["POST"])
def classify_severity():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df['HOUR'] = df['ISO_TIME'].dt.hour.fillna(0)
        df['MONTH'] = df['ISO_TIME'].dt.month.fillna(0)
        df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
        df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

        features = ['LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos']
        X = df[features].values

        if not hasattr(scaler_severity, 'mean_'):
            raise ValueError("Severity scaler is not fitted.")

        X_scaled = scaler_severity.transform(X)
        expected_features = encoder.input_shape[1]
        if X_scaled.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X_scaled.shape[1]}")

        latent_features = encoder.predict(X_scaled)
        cluster_labels = kmeans.predict(latent_features)
        severity = severity_labels.get(cluster_labels[0], "Unknown")
        return jsonify({"severity": severity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Register blueprints
app.register_blueprint(predict_earthquake_bp, url_prefix="/cyclone")
app.register_blueprint(speed_bp, url_prefix="/cyclone")
app.register_blueprint(severity_bp, url_prefix="/cyclone")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
