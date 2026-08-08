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