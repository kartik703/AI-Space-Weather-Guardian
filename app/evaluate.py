import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.models import GRUModel, TransformerModel
from app.features import PROCESSED_FILE, SCALER_X_FILE, SCALER_Y_FILE


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 12):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def inverse_scale(y_scaled: np.ndarray, scaler_y):
    y_scaled = y_scaled.reshape(-1, 1)
    return scaler_y.inverse_transform(y_scaled).flatten()


def evaluate_all():
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError("Run `python -m app.features` first to create processed data.")

    df = pd.read_csv(PROCESSED_FILE, parse_dates=["time_tag"])
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    feature_cols = [c for c in df.columns if c not in ["time_tag", "Kp", "Kp_scaled"]]
    X = df[feature_cols].values.astype(np.float32)
    y_scaled = df["Kp_scaled"].values.astype(np.float32)
    y_true_kp = df["Kp"].values.astype(np.float32)

    seq_len = 12
    X_seq, y_seq = create_sequences(X, y_scaled, seq_len=seq_len)
    y_true_kp_seq = y_true_kp[seq_len:]

    split_idx = int(0.8 * len(X_seq))
    X_val = X_seq[split_idx:]
    y_val_scaled = y_seq[split_idx:]
    y_val_kp = y_true_kp_seq[split_idx:]

    os.makedirs("models", exist_ok=True)
    results = {}

    # XGBoost
    try:
        import xgboost as xgb
        xgb_model = joblib.load("models/xgb_model.pkl")
        X_val_flat = X_val[:, -1, :]
        y_pred_scaled_xgb = xgb_model.predict(X_val_flat).astype(np.float32)
        y_pred_kp_xgb = inverse_scale(y_pred_scaled_xgb, scaler_y)
        results["XGBoost"] = {
            "MAE_Kp": float(mean_absolute_error(y_val_kp, y_pred_kp_xgb)),
            "RMSE_Kp": rmse(y_val_kp, y_pred_kp_xgb),
        }
    except Exception as e:
        print(f"XGBoost evaluation skipped: {e}")

    # GRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        gru = GRUModel(input_size=X.shape[1], hidden_size=64, num_layers=2, dropout=0.2)
        gru.load_state_dict(torch.load("models/gru_model.pth", map_location=device))
        gru.to(device).eval()
        with torch.no_grad():
            y_pred_scaled_gru = gru(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        y_pred_kp_gru = inverse_scale(y_pred_scaled_gru, scaler_y)
        results["GRU"] = {
            "MAE_Kp": float(mean_absolute_error(y_val_kp, y_pred_kp_gru)),
            "RMSE_Kp": rmse(y_val_kp, y_pred_kp_gru),
        }
    except Exception as e:
        print(f"GRU evaluation skipped: {e}")

    # Transformer
    try:
        transformer = TransformerModel(input_size=X.shape[1], d_model=64, nhead=4, num_layers=2, dropout=0.2)
        transformer.load_state_dict(torch.load("models/transformer_model.pth", map_location=device))
        transformer.to(device).eval()
        with torch.no_grad():
            y_pred_scaled_tr = transformer(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        y_pred_kp_tr = inverse_scale(y_pred_scaled_tr, scaler_y)
        results["Transformer"] = {
            "MAE_Kp": float(mean_absolute_error(y_val_kp, y_pred_kp_tr)),
            "RMSE_Kp": rmse(y_val_kp, y_pred_kp_tr),
        }
    except Exception as e:
        print(f"Transformer evaluation skipped: {e}")

    # Save metrics
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved metrics → models/metrics.json")
    print(results)

    # Plot
    # Prefer GRU predictions for plot; fallback to others
    y_pred_plot = None
    label = None
    if "GRU" in results:
        # We just computed y_pred_kp_gru above if try block succeeded
        pass
    try:
        y_pred_plot = y_pred_kp_gru
        label = "GRU"
    except:
        try:
            y_pred_plot = y_pred_kp_tr
            label = "Transformer"
        except:
            try:
                y_pred_plot = y_pred_kp_xgb
                label = "XGBoost"
            except:
                y_pred_plot = None

    if y_pred_plot is not None:
        plt.figure(figsize=(12, 5))
        plt.plot(y_val_kp, label="Actual Kp", linewidth=2)
        plt.plot(y_pred_plot, label=f"Predicted ({label})", alpha=0.85)
        plt.title(f"Actual vs Predicted Kp (Validation) — {label}")
        plt.xlabel("Time steps")
        plt.ylabel("Kp")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models/actual_vs_pred.png", dpi=150)
        print("✅ Saved plot → models/actual_vs_pred.png")

        residuals = y_val_kp - y_pred_plot
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=30, alpha=0.85)
        plt.title(f"Residuals (Actual - Predicted) — {label}")
        plt.xlabel("Kp error")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("models/residuals.png", dpi=150)
        print("✅ Saved plot → models/residuals.png")

    return results


if __name__ == "__main__":
    evaluate_all()
