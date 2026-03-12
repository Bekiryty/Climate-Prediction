from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ── CONFIG ──────────────────────────────────────────────────
REPO_ID     = "Bek54/climate-cnn-lstm-attention"
REGION_NAMES = ["Africa", "Asia", "Europe", "NorthAmerica", "SouthAmerica", "Oceania"]
SEQ_LEN     = 12  # same as training

# ── MODEL ARCHITECTURE (must match training exactly) ─────────
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size=1, cnn_filters=64, lstm_hidden=128,
                 lstm_layers=2, dropout=0.2):
        super().__init__()
        self.conv1  = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.conv2  = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool1d(2)
        self.lstm   = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers,
                              batch_first=True, dropout=dropout)
        self.attn_w = nn.Linear(lstm_hidden, 1)
        self.fc1    = nn.Linear(lstm_hidden, 64)
        self.fc2    = nn.Linear(64, 1)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)                    # (batch, features, seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # pad back to original seq length if pooled
        x = x.permute(0, 2, 1)                    # (batch, seq, features)
        out, _ = self.lstm(x)
        # Attention
        scores  = self.attn_w(out).squeeze(-1)     # (batch, seq)
        weights = torch.softmax(scores, dim=-1)
        context = (out * weights.unsqueeze(-1)).sum(1)  # (batch, hidden)
        x = self.relu(self.fc1(self.drop(context)))
        return self.fc2(x).squeeze(-1)


# ── DOWNLOAD DATA FILES FROM HF ──────────────────────────────
print("Downloading data files from HuggingFace...")
proj_path    = hf_hub_download(repo_id=REPO_ID, filename="saved_model_regional/regional_projections.json", repo_type="model")
data_path    = hf_hub_download(repo_id=REPO_ID, filename="saved_model_regional/regional_data.csv",         repo_type="model")
scalers_path = hf_hub_download(repo_id=REPO_ID, filename="saved_model_regional/regional_scalers.pkl",      repo_type="model")
print("Data files downloaded!")

# ── LOAD DATA ────────────────────────────────────────────────
with open(proj_path, "r") as f:
    regional_projections = json.load(f)

regional_df = pd.read_csv(data_path)

with open(scalers_path, "rb") as f:
    regional_scalers = pickle.load(f)

# ── LOAD MODELS FROM HF ──────────────────────────────────────
device  = torch.device("cpu")
MODELS  = {}

print("Loading regional models from Hugging Face...")
for region in REGION_NAMES:
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"saved_model_regional/{region}_model.pth",
            repo_type="model"
        )
        # Load best_params for this region if available
        try:
            params_path = hf_hub_download(
                repo_id=REPO_ID,
                filename="saved_model/best_params.pkl",
                repo_type="model"
            )
            with open(params_path, "rb") as f:
                best_params = pickle.load(f)
            model = CNNLSTMAttention(
                cnn_filters  = best_params.get("cnn_filters",  64),
                lstm_hidden  = best_params.get("lstm_hidden",  128),
                lstm_layers  = best_params.get("lstm_layers",  2),
                dropout      = best_params.get("dropout",      0.2)
            )
        except Exception:
            model = CNNLSTMAttention()

        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        MODELS[region] = model
        print(f"  ✅ {region} loaded")
    except Exception as e:
        print(f"  ⚠️  {region} failed: {e}")
        MODELS[region] = None

print("All models ready!")

# ── FALLBACK: polynomial fit ─────────────────────────────────
region_coeffs = {}
for region in REGION_NAMES:
    years  = regional_df["Year"].values
    temps  = regional_df[f"{region}_temp"].values
    region_coeffs[region] = np.polyfit(years, temps, deg=2)


def predict_with_model(region, target_year):
    """Autoregressive prediction using real CNN-LSTM model."""
    model   = MODELS.get(region)
    scaler  = regional_scalers.get(region)

    if model is None or scaler is None:
        # fallback to polynomial
        poly = np.poly1d(region_coeffs[region])
        return round(float(poly(target_year)), 2), "polynomial"

    # Build seed sequence from last SEQ_LEN known values
    temps = regional_df[f"{region}_temp"].values
    seq   = scaler.transform(temps[-SEQ_LEN:].reshape(-1, 1)).flatten().tolist()

    current_year = int(regional_df["Year"].max())
    steps = target_year - current_year

    with torch.no_grad():
        for _ in range(steps):
            x   = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out = model(x).item()
            seq.append(out)

    predicted_scaled = seq[-1]
    predicted = scaler.inverse_transform([[predicted_scaled]])[0][0]
    return round(float(predicted), 2), "cnn-lstm"


# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/projections")
def projections():
    result = {}
    for region in REGION_NAMES:
        result[region] = {
            "2030": round(regional_projections[region]["2030"], 2),
            "2050": round(regional_projections[region]["2050"], 2),
            "2100": round(regional_projections[region]["2100"], 2),
        }
    return jsonify(result)

@app.route("/historical")
def historical():
    result = {}
    for region in REGION_NAMES:
        result[region] = {
            "years": regional_df["Year"].tolist(),
            "temps": regional_df[f"{region}_temp"].tolist()
        }
    return jsonify(result)

@app.route("/summary")
def summary():
    last   = regional_df[regional_df["Year"] == 2020].iloc[0]
    result = {}
    for region in REGION_NAMES:
        result[region] = {
            "current_2020":   round(float(last[f"{region}_temp"]), 2),
            "projected_2030": round(regional_projections[region]["2030"], 2),
            "projected_2050": round(regional_projections[region]["2050"], 2),
            "projected_2100": round(regional_projections[region]["2100"], 2),
        }
    return jsonify(result)

@app.route("/predict")
def predict():
    try:
        year = int(request.args.get("year", 2050))
    except:
        return jsonify({"error": "Invalid year"}), 400

    if year < 2021 or year > 2100:
        return jsonify({"error": "Year must be between 2021 and 2100"}), 400

    result = {}
    for region in REGION_NAMES:
        predicted, method = predict_with_model(region, year)

        last        = regional_df[regional_df["Year"] == 2020].iloc[0]
        actual_2020 = round(float(last[f"{region}_temp"]), 2)
        increase    = round(predicted - actual_2020, 2)

        if predicted >= 8:   risk = "Extreme"
        elif predicted >= 5: risk = "Very High"
        elif predicted >= 3: risk = "High"
        else:                risk = "Moderate"

        result[region] = {
            "year":        year,
            "predicted":   predicted,
            "actual_2020": actual_2020,
            "increase":    increase,
            "risk":        risk,
            "method":      method   # "cnn-lstm" or "polynomial"
        }

    return jsonify({"year": year, "regions": result})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)