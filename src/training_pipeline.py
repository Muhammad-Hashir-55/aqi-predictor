"""
Training pipeline for the Pearls AQI Predictor.
Trains XGBoost, LightGBM, Random Forest, and PyTorch MLP models with 37 features.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import HF_MODEL_REPO, HF_TOKEN, TRAINING_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_COL = "target_aqi_future"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

def fetch_features_from_store() -> pd.DataFrame:
    """Fetch all features from Supabase feature store."""
    from feature_pipeline import get_db_engine
    from sqlalchemy import text
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM aqi_features ORDER BY unix_time"), conn)
    log.info("Fetched %d rows from the feature store", len(df))
    return df

def build_target(df: pd.DataFrame, horizon_hours: int = 72, tolerance_hours: int = 2) -> pd.DataFrame:
    """
    Build future target variable by merging with shifted AQI values.
    
    Args:
        df: DataFrame with features including timestamp and aqi
        horizon_hours: How many hours ahead to predict
        tolerance_hours: Tolerance for matching timestamps
    """
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
    """Select features that are present and have non-null values."""
    usable, dropped = [], []
    for col in TRAINING_FEATURE_COLS:
        if col in df.columns and df[col].notna().any():
            usable.append(col)
        else:
            dropped.append(col)
    if dropped:
        log.info("Dropping all-null feature column(s) for this run: %s", dropped)
    return usable

def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split data chronologically to avoid look-ahead bias."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def evaluate(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost regressor."""
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=500, 
        learning_rate=0.03, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, evaluate(y_test, model.predict(X_test))

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM regressor."""
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=500, 
        learning_rate=0.03, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1, 
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model, evaluate(y_test, model.predict(X_test))

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest regressor."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=600, 
        max_depth=18, 
        min_samples_leaf=2, 
        max_features=0.8, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, evaluate(y_test, model.predict(X_test))

def train_ridge(X_train, y_train, X_test, y_test):
    """Train Ridge regression."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, evaluate(y_test, model.predict(X_test))

class AQIMlp:
    """PyTorch MLP for AQI prediction."""
    def __init__(self, n_features: int, epochs: int = 500, lr: float = 5e-4):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.epochs = epochs
        self.lr = lr

    def fit(self, X_train, y_train):
        torch = self.torch
        X = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        loss_fn = torch.nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.net(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
            return self.net(X_t).view(-1).numpy()

def train_pytorch_mlp(X_train, y_train, X_test, y_test, epochs: int = 500):
    """Train PyTorch MLP with scaled targets."""
    from sklearn.preprocessing import StandardScaler
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

    model = AQIMlp(n_features=X_train.shape[1], epochs=epochs)
    model.fit(X_train, y_train_scaled)

    raw_predict = model.predict
    def predict_in_original_scale(X):
        return y_scaler.inverse_transform(raw_predict(X).reshape(-1, 1)).ravel()

    model.predict = predict_in_original_scale
    model._y_scaler = y_scaler
    return model, evaluate(y_test, model.predict(X_test))

def compute_shap_summary(model, X_train_df: pd.DataFrame, out_path: Path) -> bool:
    """Generate SHAP summary plot for feature importance visualization."""
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
        return True
    except Exception as e:
        log.warning("Could not compute SHAP summary (non-fatal): %s", e)
        return False

def push_to_model_registry(
    model_path: Path, 
    scaler_path: Path, 
    metrics_path: Path, 
    shap_path: Path | None, 
    model_name: str
) -> None:
    """Push trained model artifacts to Hugging Face Hub."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True, private=True)

    uploads = [
        (model_path, f"models/{model_name}/model.joblib"),
        (scaler_path, f"models/{model_name}/scaler.joblib"),
        (metrics_path, f"models/{model_name}/metrics.json"),
    ]
    if shap_path:
        uploads.append((shap_path, f"models/{model_name}/shap_summary.png"))
    
    for path, repo_path in uploads:
        api.upload_file(
            path_or_fileobj=str(path), 
            path_in_repo=repo_path, 
            repo_id=HF_MODEL_REPO, 
            repo_type="model"
        )
    log.info("Pushed %s to Hugging Face Hub repo %s", model_name, HF_MODEL_REPO)

def run(
    horizon_hours: int = 72, 
    test_size: float = 0.2, 
    dry_run: bool = False, 
    mlp_epochs: int = 500
) -> dict:
    """
    Main training pipeline execution.
    
    Args:
        horizon_hours: Prediction horizon in hours
        test_size: Proportion of data for testing
        dry_run: If True, skip model registry upload
        mlp_epochs: Number of epochs for MLP training
    
    Returns:
        Dictionary of all model metrics
    """
    # Fetch and prepare data
    df = fetch_features_from_store()
    if len(df) < 20:
        raise RuntimeError("Not enough rows. Run backfill_historical.py first.")

    df = build_target(df, horizon_hours=horizon_hours)
    feature_cols = select_usable_features(df)
    
    # Drop rows with missing target (but keep rows with missing features for imputation)
    df = df.dropna(subset=[TARGET_COL])

    # Split chronologically
    train_df, test_df = time_based_split(df, test_size=test_size)
    log.info("Train rows: %d, Test rows: %d, Features used: %d", 
             len(train_df), len(test_df), len(feature_cols))

    # Robust Imputation + Scaling Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    results = {}
    
    # Persistence baseline (predict current AQI as future AQI)
    baseline_metrics = evaluate(y_test, test_df["aqi"].values)
    log.info("Persistence baseline -> %s", baseline_metrics)

    # Train XGBoost
    log.info("Training XGBoost...")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results["xgboost"] = (xgb_model, xgb_metrics)
    log.info("XGBoost -> %s", xgb_metrics)

    # Train LightGBM
    log.info("Training LightGBM...")
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    results["lightgbm"] = (lgb_model, lgb_metrics)
    log.info("LightGBM -> %s", lgb_metrics)

    # Train Random Forest
    log.info("Training Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results["random_forest"] = (rf_model, rf_metrics)
    log.info("Random Forest -> %s", rf_metrics)

    # Train Ridge Regression
    log.info("Training Ridge Regression...")
    ridge_model, ridge_metrics = train_ridge(X_train, y_train, X_test, y_test)
    results["ridge"] = (ridge_model, ridge_metrics)
    log.info("Ridge -> %s", ridge_metrics)

    # Train PyTorch MLP
    log.info("Training PyTorch MLP...")
    mlp_model, mlp_metrics = train_pytorch_mlp(X_train, y_train, X_test, y_test, epochs=mlp_epochs)
    results["pytorch_mlp"] = (mlp_model, mlp_metrics)
    log.info("PyTorch MLP -> %s", mlp_metrics)

    # Select best model based on RMSE
    best_name = min(results, key=lambda name: results[name][1]["rmse"])
    best_model, best_metrics = results[best_name]
    log.info("Best model: %s (lowest RMSE) -> %s", best_name, best_metrics)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    model_path = ARTIFACTS_DIR / "model.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save all metrics
    all_metrics = {name: m for name, (_, m) in results.items()}
    all_metrics["persistence_baseline"] = baseline_metrics
    all_metrics["_selected"] = best_name
    all_metrics["_feature_cols"] = feature_cols
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    # Generate SHAP summary for tree-based models
    shap_path = None
    if best_name in ["xgboost", "lightgbm", "random_forest"]:
        shap_path = ARTIFACTS_DIR / "shap_summary.png"
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        shap_ok = compute_shap_summary(best_model, X_train_df, shap_path)
        if not shap_ok:
            shap_path = None

    # Push to model registry
    if not dry_run:
        push_to_model_registry(
            model_path, 
            scaler_path, 
            metrics_path, 
            shap_path, 
            best_name
        )

    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-hours", type=int, default=72,
                       help="Prediction horizon in hours (default: 72)")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--mlp-epochs", type=int, default=500,
                       help="Number of epochs for MLP training (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Skip model registry upload")
    args = parser.parse_args()
    run(
        horizon_hours=args.horizon_hours, 
        test_size=args.test_size, 
        dry_run=args.dry_run, 
        mlp_epochs=args.mlp_epochs
    )