from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load pre-computed projections from JSON
with open("saved_model_regional/regional_projections.json", "r") as f:
    regional_projections = json.load(f)

# Load regional data
regional_df = pd.read_csv("saved_model_regional/regional_data.csv")

region_names = ["Africa", "Asia", "Europe", "NorthAmerica", "SouthAmerica", "Oceania"]

print("Data loaded successfully!")

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/projections")
def projections():
    result = {}
    for region in region_names:
        result[region] = {
            "2030": round(regional_projections[region]["2030"], 2),
            "2050": round(regional_projections[region]["2050"], 2),
            "2100": round(regional_projections[region]["2100"], 2),
        }
    return jsonify(result)

@app.route("/historical")
def historical():
    result = {}
    for region in region_names:
        result[region] = {
            "years": regional_df["Year"].tolist(),
            "temps": regional_df[f"{region}_temp"].tolist()
        }
    return jsonify(result)

@app.route("/summary")
def summary():
    last   = regional_df[regional_df["Year"] == 2020].iloc[0]
    result = {}
    for region in region_names:
        result[region] = {
            "current_2020":   round(float(last[f"{region}_temp"]), 2),
            "projected_2030": round(regional_projections[region]["2030"], 2),
            "projected_2050": round(regional_projections[region]["2050"], 2),
            "projected_2100": round(regional_projections[region]["2100"], 2),
        }
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)