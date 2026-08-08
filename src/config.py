"""
Shared configuration for the Pearls AQI Predictor.
All secrets come from environment variables (.env locally, GitHub Secrets in CI).
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; in CI, env vars are injected directly


@dataclass(frozen=True)
class CityConfig:
    name: str
    lat: float
    lon: float


# Sialkot, Punjab, Pakistan
CITY = CityConfig(name="Sialkot", lat=32.4945, lon=74.5229)

# --- API keys / secrets (all optional at import time, validated at call time) ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")  # New: Alternative AQI data source

# --- Supabase (Postgres) feature store ---
# Use the "Transaction pooler" connection string from Supabase -> Settings -> Database
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
FEATURES_TABLE = "aqi_features"

# --- Hugging Face Hub model registry (used by training_pipeline.py) ---
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")  # e.g. "your-username/aqi-forecast-model"

# --- AQI hazard thresholds (US EPA scale, 0-500) ---
AQI_HAZARD_LEVELS = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500),
}


def classify_aqi(aqi_value: float) -> str:
    """Map a numeric AQI value to its EPA category label."""
    for label, (lo, hi) in AQI_HAZARD_LEVELS.items():
        if lo <= aqi_value <= hi:
            return label
    return "Hazardous"  # anything above 500


# --- 37-Feature Schema (Single Source of Truth) ---
DB_FEATURE_COLUMNS = [
    # Metadata
    "city", "timestamp", "unix_time",
    # Raw Pollutants & Weather
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "aqi",
    "temperature", "humidity", "pressure", "wind_speed", "wind_direction",
    # Cyclical Temporal Encodings
    "hour", "day", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Derived Ratios & Sums
    "aqi_change_rate", "pollutant_sum", "pm_ratio", "pm25_pm10_sum", "no2_o3_ratio",
    # Wind Interactions & Atmospheric Physics
    "wind_u", "wind_v", "wind_u_pm25", "wind_v_pm25",
    "temp_humidity_index", "thermal_inversion_flag",
    # Lag Features & Rolling Statistics (Memory)
    "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "pm25_rolling_mean_6h", "pm25_rolling_std_24h", "aqi_rolling_mean_24h"
]

# Exactly 37 training features (excluding metadata)
TRAINING_FEATURE_COLS = [c for c in DB_FEATURE_COLUMNS if c not in ["city", "timestamp", "unix_time"]]

