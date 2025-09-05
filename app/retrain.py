"""
Nightly retrain orchestrator:
1) Ingest latest data
2) Build features + scale
3) Train models (XGB, GRU, Transformer)
4) Evaluate & save metrics/plots
5) Choose best model and write models/model_best.json
"""

import os
import json

from app.ingestion import update_dataset
from app.features import process_and_save
from app.train import run_training
from app.evaluate import evaluate_all


def main():
    os.makedirs("models", exist_ok=True)

    print("📥 Step 1/5: Ingestion")
    update_dataset()

    print("🧱 Step 2/5: Feature engineering + scaling")
    process_and_save()

    print("🤖 Step 3/5: Train models")
    train_results = run_training()

    print("📊 Step 4/5: Evaluate & save metrics/plots")
    eval_results = evaluate_all()

    # Pick best model by lowest RMSE_Kp (Kp units) if available
    best_name = None
    best_rmse = float("inf")
    for name, m in eval_results.items():
        rm = m.get("RMSE_Kp", None)
        if rm is not None and rm < best_rmse:
            best_rmse = rm
            best_name = name

    # Fallback to training RMSE (scaled) only if eval missing
    if best_name is None:
        for name, m in train_results.items():
            rm = m.get("RMSE", None)
            if rm is not None and rm < best_rmse:
                best_rmse = rm
                best_name = name

    # Save metadata for API/dashboard
    meta_path = "models/model_best.json"
    with open(meta_path, "w") as f:
        json.dump({"best_model": best_name, "rmse_kp": float(best_rmse)}, f, indent=2)

    print(f"🏁 Best model: {best_name} (RMSE_Kp≈{best_rmse:.3f}) → {meta_path}")


if __name__ == "__main__":
    main()
