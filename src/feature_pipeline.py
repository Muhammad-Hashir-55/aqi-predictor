"""
Feature pipeline for the Pearls AQI Predictor.

1. Fetches raw weather + pollution data from OpenWeather.
2. Computes model input features (time-based, weather, pollutant, derived).
3. Writes the feature row to the Supabase (Postgres) feature store.

Run standalone:
    python src/feature_pipeline.py

Also importable — backfill_historical.py reuses fetch_air_pollution_range() / _row_from_entry().
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import CITY, OPENWEATHER_API_KEY

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


def fetch_air_pollution_range(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> dict:
    """Fetch historical air pollution data for a date range in one call.
    Used by backfill_historical.py so we don't burn one API call per hour."""
    _require_api_key()
    resp = requests.get(
        AIR_POLLUTION_HISTORY_URL,
        params={
            "lat": lat, "lon": lon,
            "start": int(start_dt.timestamp()), "end": int(end_dt.timestamp()),
            "appid": OPENWEATHER_API_KEY,
        },
        timeout=30,
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


def _row_from_entry(
    entry: dict,
    weather_json: Optional[dict] = None,
    previous_aqi: Optional[float] = None,
) -> dict:
    """Turn one raw air-pollution list entry into a flat feature dict."""
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


def build_feature_row(
    pollution_json: dict,
    weather_json: Optional[dict] = None,
    previous_aqi: Optional[float] = None,
) -> dict:
    """Turn a single-entry air-pollution API response into a flat feature dict."""
    entry = pollution_json["list"][0]
    return _row_from_entry(entry, weather_json, previous_aqi)


FEATURE_COLUMNS = [
    "city", "timestamp", "unix_time", "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3", "aqi", "aqi_change_rate",
    "hour", "day", "day_of_week", "month", "is_weekend",
    "temperature", "humidity", "pressure", "wind_speed",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aqi_features (
    city TEXT,
    timestamp TIMESTAMPTZ,
    unix_time BIGINT PRIMARY KEY,
    co DOUBLE PRECISION,
    no DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    o3 DOUBLE PRECISION,
    so2 DOUBLE PRECISION,
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    nh3 DOUBLE PRECISION,
    aqi DOUBLE PRECISION,
    aqi_change_rate DOUBLE PRECISION,
    hour INTEGER,
    day INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    is_weekend INTEGER,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION
);
"""


def get_db_engine():
    from sqlalchemy import create_engine
    from config import SUPABASE_DB_URL

    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not set. Add it to your .env or GitHub Secrets.")
    return create_engine(SUPABASE_DB_URL)


def ensure_table(engine) -> None:
    """Create the features table if it doesn't exist yet."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))


def get_previous_aqi(engine=None) -> Optional[float]:
    """Look up the most recent AQI value already in the feature store, for change-rate calc.
    Returns None if the table is empty or unreachable (first-ever run)."""
    if engine is None:
        return None
    try:
        from sqlalchemy import text
        ensure_table(engine)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT aqi FROM aqi_features ORDER BY unix_time DESC LIMIT 1")
            ).fetchone()
        return float(result[0]) if result else None
    except Exception as e:  # table may not exist yet, or connection still propagating
        log.warning("Could not fetch previous AQI (likely first run): %s", e)
        return None


def write_to_feature_store(row: dict) -> None:
    """Insert one feature row into the Supabase (Postgres) feature store."""
    write_rows_to_feature_store([row])


def write_rows_to_feature_store(rows: list) -> None:
    """Insert many feature rows in one batch — used by backfill_historical.py.
    Safe to re-run: duplicate unix_time values (from overlapping chunk boundaries,
    or re-running backfill/the hourly pipeline over an already-covered hour) are
    silently skipped rather than raising a constraint error."""
    if not rows:
        log.info("No rows to write.")
        return
    engine = get_db_engine()
    ensure_table(engine)

    df = pd.DataFrame(rows)[FEATURE_COLUMNS]
    before = len(df)
    df = df.drop_duplicates(subset="unix_time", keep="last")
    if len(df) < before:
        log.info("Dropped %d duplicate row(s) within this batch (chunk-boundary overlap)", before - len(df))

    from sqlalchemy import text
    columns = ", ".join(FEATURE_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in FEATURE_COLUMNS)
    upsert_sql = text(
        f"INSERT INTO aqi_features ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT (unix_time) DO NOTHING"
    )

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)

    log.info("Wrote %d feature rows to Supabase (%s to %s) — existing timestamps were skipped, not duplicated",
              len(records), rows[0]["timestamp"], rows[-1]["timestamp"])


def run(dry_run: bool = False) -> dict:
    """Full pipeline: fetch -> compute -> store. Returns the row that was built."""
    log.info("Fetching current air pollution + weather for %s", CITY.name)
    pollution_json = fetch_air_pollution(CITY.lat, CITY.lon)
    weather_json = fetch_weather(CITY.lat, CITY.lon)

    previous_aqi = None
    if not dry_run:
        engine = get_db_engine()
        previous_aqi = get_previous_aqi(engine)

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