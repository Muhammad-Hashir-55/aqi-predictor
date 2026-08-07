"""
Feature pipeline for the Pearls AQI Predictor.

1. Fetches raw weather + pollution data from OpenWeather.
2. Computes model input features (time-based, weather, pollutant, derived).
3. Writes the feature row to the Hopsworks Feature Store.

Run standalone:
    python src/feature_pipeline.py

Also importable — backfill_historical.py reuses fetch_air_pollution() / build_feature_row().
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import CITY, OPENWEATHER_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# OpenWeather's air pollution "aqi" field is a 1-5 index. We convert PM2.5 to a
# 0-500 US EPA AQI value instead, since that's the scale the whole project (and
# the hazard-alert thresholds) is built around.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert a PM2.5 concentration (µg/m³) to a US EPA AQI value via linear interpolation."""
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0  # cap for anything worse than the top breakpoint


def _require_api_key() -> None:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not set. Add it to your .env file or GitHub Secrets."
        )


def fetch_air_pollution(lat: float, lon: float, dt: Optional[datetime] = None) -> dict:
    """Fetch current (or, if dt given, nearest historical) air pollution data."""
    _require_api_key()
    if dt is None:
        resp = requests.get(
            AIR_POLLUTION_URL,
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY},
            timeout=15,
        )
    else:
        start_ts = int(dt.timestamp())
        end_ts = start_ts + 3600
        resp = requests.get(
            AIR_POLLUTION_HISTORY_URL,
            params={"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": OPENWEATHER_API_KEY},
            timeout=15,
        )
    resp.raise_for_status()
    return resp.json()


def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch current weather. (OpenWeather's free tier has no historical weather endpoint,
    so backfill only carries pollution + time features — see backfill_historical.py.)"""
    _require_api_key()
    resp = requests.get(
        WEATHER_URL,
        params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def build_feature_row(
    pollution_json: dict,
    weather_json: Optional[dict] = None,
    previous_aqi: Optional[float] = None,
) -> dict:
    """Turn raw API payloads into a flat feature dict ready for the feature store."""
    entry = pollution_json["list"][0]
    components = entry["components"]
    ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)

    pm25 = components.get("pm2_5", 0.0)
    aqi = pm25_to_aqi(pm25)

    row = {
        "city": CITY.name,
        "timestamp": ts,
        "unix_time": entry["dt"],
        # pollutant components
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": pm25,
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
        # target / derived
        "aqi": aqi,
        "aqi_change_rate": (aqi - previous_aqi) if previous_aqi is not None else 0.0,
        # time-based features
        "hour": ts.hour,
        "day": ts.day,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
    }

    if weather_json is not None:
        row.update(
            {
                "temperature": weather_json.get("main", {}).get("temp"),
                "humidity": weather_json.get("main", {}).get("humidity"),
                "pressure": weather_json.get("main", {}).get("pressure"),
                "wind_speed": weather_json.get("wind", {}).get("speed"),
            }
        )
    else:
        row.update({"temperature": None, "humidity": None, "pressure": None, "wind_speed": None})

    return row


def get_previous_aqi(fs_project=None) -> Optional[float]:
    """Look up the most recent AQI value already in the feature store, for change-rate calc.
    Returns None if the feature group is empty or unreachable (first-ever run)."""
    if fs_project is None:
        return None
    try:
        fg = fs_project.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        if df.empty:
            return None
        return float(df.sort_values("unix_time").iloc[-1]["aqi"])
    except Exception as e:  # feature group may not exist yet
        log.warning("Could not fetch previous AQI (likely first run): %s", e)
        return None


def write_to_feature_store(row: dict) -> None:
    """Insert one feature row into the Hopsworks feature store."""
    import hopsworks
    from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is not set. Add it to your .env or GitHub Secrets.")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()

    df = pd.DataFrame([row])

    try:
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    except Exception:
        fg = fs.create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description="Hourly AQI + weather features for Sialkot",
            primary_key=["city", "unix_time"],
            event_time="timestamp",
            online_enabled=True,
        )
    fg.insert(df, write_options={"wait_for_job": True})
    log.info("Inserted feature row for %s at %s", row["city"], row["timestamp"])


def run(dry_run: bool = False) -> dict:
    """Full pipeline: fetch -> compute -> store. Returns the row that was built."""
    log.info("Fetching current air pollution + weather for %s", CITY.name)
    pollution_json = fetch_air_pollution(CITY.lat, CITY.lon)
    weather_json = fetch_weather(CITY.lat, CITY.lon)

    previous_aqi = None
    project = None
    if not dry_run:
        import hopsworks
        from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT
        if HOPSWORKS_API_KEY:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
            previous_aqi = get_previous_aqi(project.get_feature_store() if project else None)

    row = build_feature_row(pollution_json, weather_json, previous_aqi)
    log.info("Computed feature row: AQI=%.1f, PM2.5=%.1f", row["aqi"], row["pm2_5"])

    if dry_run:
        log.info("Dry run — skipping feature store write.")
    else:
        write_to_feature_store(row)

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AQI feature pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + compute only, skip Hopsworks write")
    args = parser.parse_args()
    run(dry_run=args.dry_run)