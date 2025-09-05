import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from app.models import GRUModel, TransformerModel
from app.features import PROCESSED_FILE


# --------------------------
# Helpers
# --------------------------
def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 12):
    """Build sliding window sequences for sequence models."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def rmse(y_true, y_pred):
    """Compatibility RMSE for older scikit-learn (no squared=False)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    patience: int = 6,
):
    """Generic PyTorch training loop with early stopping on RMSE."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
            pv = model(Xv).cpu().numpy().flatten()
            cur_rmse = rmse(y_val, pv)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{model.__class__.__name__}] Epoch {epoch}/{epochs}  RMSE: {cur_rmse:.4f}")

        if cur_rmse + 1e-9 < best_rmse:
            best_rmse = cur_rmse
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch} (best RMSE: {best_rmse:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final preds after loading best state
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
        pv = model(Xv).cpu().numpy().flatten()
    mae = float(mean_absolute_error(y_val, pv))
    final_rmse = rmse(y_val, pv)
    return model, pv, mae, final_rmse


# --------------------------
# Main training pipeline
# --------------------------
def run_training():
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_FILE}. "
            "Run `python -m app.features` first."
        )

    # Load processed data (already scaled features + 'Kp_scaled' target)
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["time_tag"])

    # Features = all except timestamps and unscaled/target columns
    feature_cols = [c for c in df.columns if c not in ["time_tag", "Kp", "Kp_scaled"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Kp_scaled"].values.astype(np.float32)

    # Build sequences for GRU/Transformer
    seq_len = 12
    X_seq, y_seq = create_sequences(X, y, seq_len=seq_len)

    # Time-ordered split (no shuffling for time series)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    results = {}
    os.makedirs("models", exist_ok=True)

    # ================== XGBoost (baseline) ==================
    print("\n🚀 Training XGBoost... (CPU, no early stopping for compatibility)")
    # Use only the last timestep features for the tabular model
    Xtr_flat, Xval_flat = X_train[:, -1, :], X_val[:, -1, :]

    # Compatible across old/new versions (CPU)
    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
    )

    # Older versions accept eval_set but not early stopping/callbacks
    try:
        xgb_model.fit(Xtr_flat, y_train, eval_set=[(Xval_flat, y_val)], verbose=False)
    except TypeError:
        # fall back if this xgboost doesn't accept eval_set/verbose signature
        xgb_model.fit(Xtr_flat, y_train)

    xgb_preds = xgb_model.predict(Xval_flat)
    xgb_mae = float(mean_absolute_error(y_val, xgb_preds))
    xgb_rmse = rmse(y_val, xgb_preds)
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    print(f"✅ XGBoost → MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")
    results["XGBoost"] = {"MAE": xgb_mae, "RMSE": xgb_rmse}

    # ================== GRU (sequence) ==================
    print("\n🚀 Training GRU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru = GRUModel(input_size=X.shape[1], hidden_size=64, num_layers=2, dropout=0.2).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    gru, gru_preds, gru_mae, gru_rmse = train_torch_model(
        gru, train_loader, X_val, y_val, device, epochs=80, patience=8
    )
    torch.save(gru.state_dict(), "models/gru_model.pth")
    print(f"✅ GRU → MAE: {gru_mae:.4f}, RMSE: {gru_rmse:.4f}")
    results["GRU"] = {"MAE": gru_mae, "RMSE": gru_rmse}

    # ================== Transformer (sequence) ==================
    print("\n🚀 Training Transformer...")
    transformer = TransformerModel(
        input_size=X.shape[1], d_model=64, nhead=4, num_layers=2, dropout=0.2
    ).to(device)

    transformer, tr_preds, tr_mae, tr_rmse = train_torch_model(
        transformer, train_loader, X_val, y_val, device, epochs=80, patience=8
    )
    torch.save(transformer.state_dict(), "models/transformer_model.pth")
    print(f"✅ Transformer → MAE: {tr_mae:.4f}, RMSE: {tr_rmse:.4f}")
    results["Transformer"] = {"MAE": tr_mae, "RMSE": tr_rmse}

    print("\n📊 Final Comparison:", results)
    return results


if __name__ == "__main__":
    run_training()
