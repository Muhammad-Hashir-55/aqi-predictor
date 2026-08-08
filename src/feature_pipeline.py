"""
Feature pipeline for the Pearls AQI Predictor with 37 engineered features.
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import CITY, OPENWEATHER_API_KEY, DB_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

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
    """Convert PM2.5 concentration to AQI using EPA breakpoints."""
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0

def _require_api_key() -> None:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set. Add it to your .env file.")

def fetch_air_pollution(lat: float, lon: float, dt: Optional[datetime] = None) -> dict:
    """
    Fetch current or historical air pollution data.
    If dt is None, fetches current data. Otherwise fetches data for the given hour.
    """
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
    """Fetch air pollution data over a time range."""
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
    """Fetch current weather data."""
    _require_api_key()
    resp = requests.get(
        WEATHER_URL,
        params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def engineer_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized transformer that computes all 37 features across a time-series DataFrame.
    """
    df = df.sort_values("timestamp").copy()
    
    # Temporal cyclical encodings
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # Ratios and Sums
    df["pollutant_sum"] = df[["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]].sum(axis=1)
    df["pm_ratio"] = df["pm2_5"] / df["pm10"].clip(lower=1e-6)
    df["pm25_pm10_sum"] = df["pm2_5"] + df["pm10"]
    df["no2_o3_ratio"] = df["no2"] / df["o3"].clip(lower=1e-6)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)

    # Wind Vector Interactions
    wind_rad = np.radians(df["wind_direction"].fillna(0))
    df["wind_u"] = -df["wind_speed"] * np.sin(wind_rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(wind_rad)
    df["wind_u_pm25"] = df["wind_u"] * df["pm2_5"]
    df["wind_v_pm25"] = df["wind_v"] * df["pm2_5"]

    # Atmospheric Physics
    df["temp_humidity_index"] = df["temperature"] - (0.55 - 0.0055 * df["humidity"]) * (df["temperature"] - 14.5)
    df["thermal_inversion_flag"] = ((df["wind_speed"] < 2.0) & (df["pressure"] > 1015) & (df["temperature"] < 15)).astype(int)

    # Memory: Lags & Rolling Statistics
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_12h"] = df["aqi"].shift(12)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["pm25_rolling_mean_6h"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm25_rolling_std_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).std().fillna(0)
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    return df

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
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    hour INTEGER,
    day INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    is_weekend INTEGER,
    hour_sin DOUBLE PRECISION,
    hour_cos DOUBLE PRECISION,
    dow_sin DOUBLE PRECISION,
    dow_cos DOUBLE PRECISION,
    month_sin DOUBLE PRECISION,
    month_cos DOUBLE PRECISION,
    aqi_change_rate DOUBLE PRECISION,
    pollutant_sum DOUBLE PRECISION,
    pm_ratio DOUBLE PRECISION,
    pm25_pm10_sum DOUBLE PRECISION,
    no2_o3_ratio DOUBLE PRECISION,
    wind_u DOUBLE PRECISION,
    wind_v DOUBLE PRECISION,
    wind_u_pm25 DOUBLE PRECISION,
    wind_v_pm25 DOUBLE PRECISION,
    temp_humidity_index DOUBLE PRECISION,
    thermal_inversion_flag INTEGER,
    aqi_lag_6h DOUBLE PRECISION,
    aqi_lag_12h DOUBLE PRECISION,
    aqi_lag_24h DOUBLE PRECISION,
    pm25_rolling_mean_6h DOUBLE PRECISION,
    pm25_rolling_std_24h DOUBLE PRECISION,
    aqi_rolling_mean_24h DOUBLE PRECISION
);
"""

def get_db_engine():
    from sqlalchemy import create_engine
    from config import SUPABASE_DB_URL
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not set.")
    return create_engine(SUPABASE_DB_URL)

def ensure_table(engine) -> None:
    """Create the feature table if it doesn't exist."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    log.info("Verified table exists.")

def drop_table(engine) -> None:
    """Drop the feature table completely (use with caution)."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS aqi_features CASCADE;"))
    log.info("Dropped existing aqi_features table.")

def get_previous_aqi(engine) -> Optional[float]:
    """Get the most recent AQI value from the feature store."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT aqi FROM aqi_features ORDER BY unix_time DESC LIMIT 1")).fetchone()
        return float(result[0]) if result else None
    except Exception as e:
        log.warning("Could not fetch previous AQI: %s", e)
        return None

def write_rows_to_feature_store(rows: list, batch_size: int = 100) -> None:
    """Write feature rows to Supabase with upsert support."""
    if not rows:
        return
    
    engine = get_db_engine()
    ensure_table(engine)

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="unix_time", keep="last")
    if len(df) < before:
        log.info("Dropped %d duplicate row(s)", before - len(df))

    from sqlalchemy import text
    columns = ", ".join(DB_FEATURE_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in DB_FEATURE_COLUMNS)
    upsert_sql = text(f"INSERT INTO aqi_features ({columns}) VALUES ({placeholders}) ON CONFLICT (unix_time) DO NOTHING")

    records = df.to_dict(orient="records")
    total_rows = len(records)
    
    # Insert in batches to avoid server timeout
    with engine.begin() as conn:
        for i in range(0, total_rows, batch_size):
            batch = records[i:i + batch_size]
            conn.execute(upsert_sql, batch)
            log.info("Inserted batch %d/%d (%d rows)", 
                    (i // batch_size) + 1, 
                    (total_rows + batch_size - 1) // batch_size,
                    len(batch))

    log.info("Wrote %d feature rows to Supabase", total_rows)

def build_feature_row(pollution_json: dict, weather_json: Optional[dict] = None, previous_aqi: Optional[float] = None) -> dict:
    """
    Build a single feature row from pollution and weather data.
    Preserves the original function signature for backward compatibility.
    """
    entry = pollution_json["list"][0]
    components = entry["components"]
    ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    pm25 = components.get("pm2_5", 0.0)
    aqi = pm25_to_aqi(pm25)

    row = {
        "city": CITY.name,
        "timestamp": ts,
        "unix_time": entry["dt"],
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": pm25,
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
        "aqi": aqi,
        "aqi_change_rate": (aqi - previous_aqi) if previous_aqi is not None else 0.0,
        "pollutant_sum": sum(
            float(components.get(name) or 0.0)
            for name in ("co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3")
        ),
        "pm_ratio": (float(components.get("pm2_5") or 0.0) / max(float(components.get("pm10") or 0.0), 1e-6)),
        "pm25_pm10_sum": (float(components.get("pm2_5") or 0.0) + float(components.get("pm10") or 0.0)),
        "no2_o3_ratio": (float(components.get("no2") or 0.0) / max(float(components.get("o3") or 0.0), 1e-6)),
        "hour": ts.hour,
        "day": ts.day,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
        "hour_sin": math.sin(2 * math.pi * ts.hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * ts.hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * ts.weekday() / 7.0),
        "dow_cos": math.cos(2 * math.pi * ts.weekday() / 7.0),
        "month_sin": math.sin(2 * math.pi * (ts.month - 1) / 12.0),
        "month_cos": math.cos(2 * math.pi * (ts.month - 1) / 12.0),
        # New 37-feature fields (set to defaults for backward compatibility)
        "wind_direction": weather_json.get("wind", {}).get("deg", 0.0) if weather_json else 0.0,
        "wind_u": 0.0,
        "wind_v": 0.0,
        "wind_u_pm25": 0.0,
        "wind_v_pm25": 0.0,
        "temp_humidity_index": 0.0,
        "thermal_inversion_flag": 0,
        "aqi_lag_6h": None,
        "aqi_lag_12h": None,
        "aqi_lag_24h": None,
        "pm25_rolling_mean_6h": 0.0,
        "pm25_rolling_std_24h": 0.0,
        "aqi_rolling_mean_24h": 0.0,
    }

    if weather_json is not None:
        row.update({
            "temperature": weather_json.get("main", {}).get("temp"),
            "humidity": weather_json.get("main", {}).get("humidity"),
            "pressure": weather_json.get("main", {}).get("pressure"),
            "wind_speed": weather_json.get("wind", {}).get("speed"),
            "wind_direction": weather_json.get("wind", {}).get("deg", 0.0),
        })

    return row

def run(dry_run: bool = False) -> dict:
    """
    Main execution function. Fetches current data, engineers features, and stores them.
    
    Args:
        dry_run: If True, don't write to database.
    
    Returns:
        The feature row dictionary.
    """
    log.info("Fetching current air pollution + weather for %s", CITY.name)
    pollution_json = fetch_air_pollution(CITY.lat, CITY.lon)
    weather_json = fetch_weather(CITY.lat, CITY.lon)
    
    # Build initial feature row
    engine = get_db_engine() if not dry_run else None
    previous_aqi = get_previous_aqi(engine) if engine else None
    row = build_feature_row(pollution_json, weather_json, previous_aqi)
    
    # If we have historical data, compute advanced features using the vectorized approach
    if engine:
        try:
            ensure_table(engine)
            # Fetch last 30 hours from DB for lag & rolling stats
            with engine.connect() as conn:
                historical_df = pd.read_sql(
                    "SELECT * FROM aqi_features ORDER BY unix_time DESC LIMIT 30", 
                    conn
                )
            
            if not historical_df.empty:
                # Combine with new row
                combined_df = pd.concat([historical_df, pd.DataFrame([row])], ignore_index=True)
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                
                # Engineer all 37 features
                engineered_df = engineer_features_dataframe(combined_df)
                row = engineered_df.iloc[-1].to_dict()
                log.info("Computed 37 features using historical context")
        except Exception as e:
            log.warning("Could not compute advanced features: %s", e)
    
    log.info("Computed feature row: AQI=%.1f, PM2.5=%.1f", row["aqi"], row["pm2_5"])
    
    if not dry_run:
        write_rows_to_feature_store([row])
    
    return row

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)