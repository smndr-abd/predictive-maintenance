"""
app.py
------
Flask deployment server for the Predictive Maintenance (RUL) dashboard.

Serves:
  GET  /                        -> dashboard HTML
  GET  /api/engines             -> list of test-set engine IDs + true RUL
  GET  /api/engine/<id>         -> full sensor trace + prediction for one engine
  GET  /api/predict_all         -> predictions for every test engine (both models)
  POST /api/predict             -> predict RUL from a raw JSON sensor window

Run:
    python app.py
Then open http://localhost:5000
"""

import json
import os
import pickle
import sys

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from models import build_model  # noqa: E402
from preprocessing import (  # noqa: E402
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    add_train_rul,
    apply_scaler,
    load_raw,
    load_rul,
    make_test_sequences,
    RUL_CAP,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CONFIG = json.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

N_FEATURES = CONFIG["n_features"]

MODELS = {}
for name, filename in [("lstm", "lstm_best.pt"), ("transformer", "transformer_best.pt")]:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        m = build_model(name, N_FEATURES)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        MODELS[name] = m

print(f"Loaded models: {list(MODELS.keys())}")

# ---------------------------------------------------------------------------
# Pre-load the FD001 test set (used to populate the dashboard)
# ---------------------------------------------------------------------------
test_df_raw = load_raw(os.path.join(DATA_DIR, "test_FD001.txt"))
test_rul_true = load_rul(os.path.join(DATA_DIR, "RUL_FD001.txt"))
test_df_scaled = apply_scaler(test_df_raw, SCALER)


def get_engine_ids():
    return sorted(test_df_raw["unit_number"].unique().tolist())


@torch.no_grad()
def predict_rul(model_name: str, X: np.ndarray) -> np.ndarray:
    """X: (batch, seq_len, n_features) already scaled."""
    model = MODELS[model_name]
    xb = torch.tensor(X, dtype=torch.float32)
    preds = model(xb).numpy()
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", models=list(MODELS.keys()))


@app.route("/api/engines")
def api_engines():
    ids = get_engine_ids()
    rows = []
    for i, engine_id in enumerate(ids):
        rows.append({
            "engine_id": int(engine_id),
            "true_rul": float(np.clip(test_rul_true[i], 0, RUL_CAP)),
        })
    return jsonify(rows)


@app.route("/api/engine/<int:engine_id>")
def api_engine_detail(engine_id):
    model_name = request.args.get("model", "lstm")
    if model_name not in MODELS:
        return jsonify({"error": f"model '{model_name}' not available"}), 400

    unit_df = test_df_raw[test_df_raw["unit_number"] == engine_id].sort_values("time_cycles")
    if unit_df.empty:
        return jsonify({"error": "engine not found"}), 404

    unit_scaled = test_df_scaled[test_df_scaled["unit_number"] == engine_id].sort_values("time_cycles")
    X = make_test_sequences(unit_scaled, SEQUENCE_LENGTH)  # shape (1, seq_len, n_features)
    pred = predict_rul(model_name, X)[0]

    ids = get_engine_ids()
    idx = ids.index(engine_id)
    true_rul = float(np.clip(test_rul_true[idx], 0, RUL_CAP))

    # A few representative raw sensor traces for the chart (unscaled, human-readable)
    trace_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_15"]
    traces = {
        s: unit_df[s].tolist() for s in trace_sensors
    }

    status = "critical" if pred < 20 else "warning" if pred < 50 else "healthy"

    return jsonify({
        "engine_id": engine_id,
        "cycles": unit_df["time_cycles"].tolist(),
        "sensor_traces": traces,
        "predicted_rul": round(float(pred), 1),
        "true_rul": true_rul,
        "status": status,
        "model_used": model_name,
    })


@app.route("/api/predict_all")
def api_predict_all():
    model_name = request.args.get("model", "lstm")
    if model_name not in MODELS:
        return jsonify({"error": f"model '{model_name}' not available"}), 400

    ids = get_engine_ids()
    results = []
    for i, engine_id in enumerate(ids):
        unit_scaled = test_df_scaled[test_df_scaled["unit_number"] == engine_id].sort_values("time_cycles")
        X = make_test_sequences(unit_scaled, SEQUENCE_LENGTH)
        pred = predict_rul(model_name, X)[0]
        true_rul = float(np.clip(test_rul_true[i], 0, RUL_CAP))
        status = "critical" if pred < 20 else "warning" if pred < 50 else "healthy"
        results.append({
            "engine_id": int(engine_id),
            "predicted_rul": round(float(pred), 1),
            "true_rul": true_rul,
            "status": status,
        })
    return jsonify(results)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict RUL from a raw sensor window POSTed as JSON:
    { "model": "lstm", "window": [[<17 features>], ... 30 rows ...] }
    Rows must already be in FEATURE_COLS order and RAW (unscaled) units;
    this endpoint applies the same scaler used during training.
    """
    payload = request.get_json(force=True)
    model_name = payload.get("model", "lstm")
    window = payload.get("window")

    if model_name not in MODELS:
        return jsonify({"error": f"model '{model_name}' not available"}), 400
    if window is None:
        return jsonify({"error": "missing 'window' field"}), 400

    arr = np.array(window, dtype=np.float32)
    if arr.shape != (SEQUENCE_LENGTH, N_FEATURES):
        return jsonify({
            "error": f"expected window shape ({SEQUENCE_LENGTH}, {N_FEATURES}), got {arr.shape}",
            "expected_feature_order": FEATURE_COLS,
        }), 400

    import pandas as pd
    scaled = SCALER.transform(pd.DataFrame(arr, columns=FEATURE_COLS))
    pred = predict_rul(model_name, scaled[np.newaxis, ...])[0]
    return jsonify({"predicted_rul": round(float(pred), 1), "model_used": model_name})


@app.route("/api/models")
def api_models():
    return jsonify({"available_models": list(MODELS.keys())})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
