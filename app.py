from flask import Flask, jsonify
from flask_cors import CORS
# Import blueprints from your folders
from cyclone.routes import cyclone_
from combined.routes import speed_bp
from combined.routes import severity_bp
from combined.routes import predict_earthquake_bp
  # Uncomment if needed

import torch.nn as nn
import torch
# from server.fetcher import fetcher_bp  # Uncomment if needed
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

        
autoencoder = Autoencoder(input_dim=4, hidden_dim=16, latent_dim=64)
# Load the saved model weights (state_dict)
# Use weights_only=False to load the entire model

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(speed_bp, url_prefix='/combined')
app.register_blueprint(severity_bp, url_prefix='/combined')
app.register_blueprint(predict_earthquake_bp, url_prefix='/combined')  # Uncomment if needed
app.register_blueprint(cyclone_, url_prefix='/cyclone')  # Uncomment if needed

# app.register_blueprint(fetcher_bp, url_prefix='/server')

@app.route('/')
def home():
    return jsonify({"message": "Unified Disaster Prediction API is running."})

if __name__ == '__main__':
    app.run(port=5000)
