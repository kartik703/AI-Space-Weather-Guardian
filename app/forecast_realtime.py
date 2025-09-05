import os
import json
import numpy as np
import pandas as pd
import joblib
import torch

from app.models import GRUModel
from app.features import build_features, SCALER_X_FILE, SCALER_Y_FILE


def load_latest_processed_rows(raw_csv="data/raw_data.csv", max_lag=12):
    if not os.path.exists(raw_csv):
        raise FileNotFoundError("No raw data found. Run ingestion first.")
    df = pd.read_csv(raw_csv, parse_dates=["time_tag"])
    df = build_features(df, target="Kp", max_lag=max_lag)
    return df


def _pick_best_model():
    meta_path = "models/model_best.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            return meta.get("best_model", "GRU"), float(meta.get("rmse_kp", 0.3))
    # default
    return "GRU", 0.3


def forecast_next_kp(horizons=(1, 3, 6), seq_len=12):
    best_model, rmse_kp = _pick_best_model()

    # Load latest engineered features
    df = load_latest_processed_rows()
    feature_cols = [c for c in df.columns if c not in ["time_tag", "Kp"]]

    # Load scalers
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    X_raw = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler_X.transform(X_raw)

    # Build most recent sequence
    if len(X_scaled) < seq_len:
        raise ValueError(f"Not enough data to build a sequence of length {seq_len}.")
    current_seq = X_scaled[-seq_len:].copy()

    # Load model (for now, use GRU path; extend here for XGB/Transformer if desired)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_scaled.shape[1]

    model = GRUModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load("models/gru_model.pth", map_location=device))
    model.to(device).eval()

    forecasts_scaled = []
    with torch.no_grad():
        for _ in horizons:
            x = torch.tensor(current_seq[np.newaxis, ...], dtype=torch.float32).to(device)
            yhat_scaled = model(x).cpu().numpy().flatten()[0]
            forecasts_scaled.append(yhat_scaled)

            # Simple roll-forward of sequence window
            next_vec = current_seq[-1].copy()
            current_seq = np.vstack([current_seq[1:], next_vec])

    # Inverse-scale to real Kp
    kp_vals = joblib.load(SCALER_Y_FILE).inverse_transform(
        np.array(forecasts_scaled).reshape(-1, 1)
    ).flatten()

    # Return dict with ±rmse_kp band
    out = {f"{h}h": float(v) for h, v in zip(horizons, kp_vals)}
    return {"forecasts": out, "model": best_model, "uncertainty_kp": float(rmse_kp)}


if __name__ == "__main__":
    preds = forecast_next_kp()
    print("🔮 Real-time Kp forecast:", preds)
