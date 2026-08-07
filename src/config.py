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
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")

# --- Feature store naming ---
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
MODEL_NAME = "aqi_forecast_model"

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