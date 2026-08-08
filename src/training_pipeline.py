"""
Training pipeline for the Pearls AQI Predictor.

1. Fetches historical features from the Supabase feature store.
2. Builds the target: actual AQI ~72 hours (3 days) after each row's timestamp.
3. Trains and evaluates Random Forest, Ridge Regression, and a small PyTorch MLP.
4. Computes SHAP feature-importance for the tree model.
5. Pushes the best model (+ scaler + metrics + SHAP plot) to the Hugging Face Hub.

Run:
    python src/training_pipeline.py
    python src/training_pipeline.py --horizon-hours 72 --dry-run   # skip the HF push
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import HF_MODEL_REPO, HF_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_FEATURE_COLS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "aqi", "aqi_change_rate",
    "hour", "day", "day_of_week", "month", "is_weekend",
    "temperature", "humidity", "pressure", "wind_speed",
]
TARGET_COL = "target_aqi_future"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def fetch_features_from_store() -> pd.DataFrame:
    """Pull every row out of the Supabase feature store, oldest first."""
    from feature_pipeline import get_db_engine
    from sqlalchemy import text

    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM aqi_features ORDER BY unix_time"), conn)
    log.info("Fetched %d rows from the feature store", len(df))
    return df


def build_target(df: pd.DataFrame, horizon_hours: int = 72, tolerance_hours: int = 2) -> pd.DataFrame:
    """Add a target column: the actual AQI ~horizon_hours after each row's timestamp.
    Rows near the end of the dataset (where no future row exists yet) are dropped —
    we can't train on a label we don't have."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    future = df[["timestamp", "aqi"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(hours=horizon_hours)
    future = future.rename(columns={"aqi": TARGET_COL}).sort_values("timestamp")

    merged = pd.merge_asof(
        df, future, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )
    before = len(merged)
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    log.info("Built %d-hour-ahead target: %d/%d rows have a matching future label",
              horizon_hours, len(merged), before)
    return merged


def select_usable_features(df: pd.DataFrame) -> list:
    """Drop candidate feature columns that are entirely null (e.g. weather columns
    when the dataset is still all-backfill, since OpenWeather's free tier has no
    historical weather). Once the hourly pipeline accumulates enough live data,
    rerunning this will automatically pick weather features back up."""
    usable, dropped = [], []
    for col in CANDIDATE_FEATURE_COLS:
        if col in df.columns and df[col].notna().any():
            usable.append(col)
        else:
            dropped.append(col)
    if dropped:
        log.info("Dropping all-null feature column(s) for this run: %s", dropped)
    return usable


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split — the last test_size fraction (by time) is the test set.
    Random shuffling would leak future information into training for a forecast task."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_random_forest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    return model, metrics


def train_ridge(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    return model, metrics


class AQIMlp:
    """Thin wrapper so the PyTorch model exposes .predict() like the sklearn ones,
    keeping evaluate() and the model-comparison logic uniform across all three."""

    def __init__(self, n_features: int, epochs: int = 100, lr: float = 1e-3):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.epochs = epochs
        self.lr = lr

    def fit(self, X_train, y_train):
        torch = self.torch
        X = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.net(X), y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                log.info("  MLP epoch %d/%d — loss %.3f", epoch + 1, self.epochs, loss.item())
        return self

    def predict(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
            return self.net(X_t).view(-1).numpy()


def train_pytorch_mlp(X_train, y_train, X_test, y_test, epochs: int = 100):
    """Target values (AQI, ~0-500 scale) get standardized before training — an
    untrained net starts near output 0, and without this it takes far too many
    gradient steps to reach the right scale, which was tanking accuracy."""
    from sklearn.preprocessing import StandardScaler

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

    model = AQIMlp(n_features=X_train.shape[1], epochs=epochs)
    model.fit(X_train, y_train_scaled)

    raw_predict = model.predict

    def predict_in_original_scale(X):
        scaled_preds = raw_predict(X)
        return y_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()

    model.predict = predict_in_original_scale  # so callers always get real AQI values back
    model._y_scaler = y_scaler

    metrics = evaluate(y_test, model.predict(X_test))
    return model, metrics


def compute_shap_summary(model, X_train_df: pd.DataFrame, out_path: Path) -> bool:
    """SHAP feature-importance plot for the tree model. Returns True on success —
    SHAP only supports certain model types, so this is skipped gracefully otherwise."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        sample = X_train_df.sample(min(200, len(X_train_df)), random_state=42)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved SHAP summary plot to %s", out_path)
        return True
    except Exception as e:
        log.warning("Could not compute SHAP summary (non-fatal, continuing): %s", e)
        return False


def push_to_model_registry(model_path: Path, scaler_path: Path, metrics_path: Path,
                            shap_path: Path | None, model_name: str) -> None:
    """Upload the winning model + its artifacts to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    if not HF_TOKEN or not HF_MODEL_REPO:
        raise RuntimeError("HF_TOKEN / HF_MODEL_REPO not set. Add them to your .env or GitHub Secrets.")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True, private=True)

    for path, repo_path in [
        (model_path, f"models/{model_name}/model.joblib"),
        (scaler_path, f"models/{model_name}/scaler.joblib"),
        (metrics_path, f"models/{model_name}/metrics.json"),
    ] + ([(shap_path, f"models/{model_name}/shap_summary.png")] if shap_path else []):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
    log.info("Pushed %s to Hugging Face Hub repo %s", model_name, HF_MODEL_REPO)


def run(horizon_hours: int = 72, test_size: float = 0.2, dry_run: bool = False, mlp_epochs: int = 100) -> dict:
    df = fetch_features_from_store()
    if len(df) < 20:
        raise RuntimeError(f"Only {len(df)} rows in the feature store — run backfill_historical.py first.")

    df = build_target(df, horizon_hours=horizon_hours)
    feature_cols = select_usable_features(df)
    df = df.dropna(subset=feature_cols + [TARGET_COL])

    train_df, test_df = time_based_split(df, test_size=test_size)
    log.info("Train rows: %d, Test rows: %d, Features used: %s", len(train_df), len(test_df), feature_cols)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    results = {}
    log.info("Evaluating naive persistence baseline (predict future AQI = current AQI)...")
    baseline_metrics = evaluate(y_test, test_df["aqi"].values)
    log.info("Persistence baseline -> %s", baseline_metrics)

    log.info("Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = (rf_model, rf_metrics)
    log.info("Random Forest -> %s", rf_metrics)

    log.info("Training Ridge Regression...")
    ridge_model, ridge_metrics = train_ridge(X_train, y_train, X_test, y_test)
    results["ridge"] = (ridge_model, ridge_metrics)
    log.info("Ridge -> %s", ridge_metrics)

    log.info("Training PyTorch MLP...")
    mlp_model, mlp_metrics = train_pytorch_mlp(X_train, y_train, X_test, y_test, epochs=mlp_epochs)
    results["pytorch_mlp"] = (mlp_model, mlp_metrics)
    log.info("PyTorch MLP -> %s", mlp_metrics)

    best_name = min(results, key=lambda name: results[name][1]["rmse"])
    best_model, best_metrics = results[best_name]
    log.info("Best model: %s (lowest RMSE) -> %s", best_name, best_metrics)

    if best_metrics["rmse"] >= baseline_metrics["rmse"]:
        log.warning(
            "None of the trained models beat the naive persistence baseline (RMSE %.2f). "
            "This usually means the training window is too short or too different from the "
            "test period — more backfilled history (try --days 90) is the most direct fix.",
            baseline_metrics["rmse"],
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = ARTIFACTS_DIR / "model.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    all_metrics = {name: m for name, (_, m) in results.items()}
    all_metrics["persistence_baseline"] = baseline_metrics
    all_metrics["_selected"] = best_name
    all_metrics["_feature_cols"] = feature_cols
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    shap_path = ARTIFACTS_DIR / "shap_summary.png"
    shap_ok = False
    if best_name == "random_forest":
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        shap_ok = compute_shap_summary(best_model, X_train_df, shap_path)

    if dry_run:
        log.info("Dry run — skipping Hugging Face Hub push. Artifacts saved locally in %s", ARTIFACTS_DIR)
    else:
        push_to_model_registry(model_path, scaler_path, metrics_path,
                                shap_path if shap_ok else None, best_name)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register the AQI forecast model")
    parser.add_argument("--horizon-hours", type=int, default=72, help="Forecast horizon in hours (default 72 = 3 days)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data (chronologically last) held out for testing")
    parser.add_argument("--mlp-epochs", type=int, default=100, help="Training epochs for the PyTorch MLP")
    parser.add_argument("--dry-run", action="store_true", help="Train + evaluate only, skip the Hugging Face Hub push")
    args = parser.parse_args()
    run(horizon_hours=args.horizon_hours, test_size=args.test_size, dry_run=args.dry_run, mlp_epochs=args.mlp_epochs)