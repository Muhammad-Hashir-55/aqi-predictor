"""
Feature pipeline for the Pearls AQI Predictor.
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone
from typing import Optional

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
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0

def _require_api_key() -> None:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set. Add it to your .env file.")

def fetch_air_pollution(lat: float, lon: float, dt: Optional[datetime] = None) -> dict:
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
    }

    if weather_json is not None:
        row.update({
            "temperature": weather_json.get("main", {}).get("temp"),
            "humidity": weather_json.get("main", {}).get("humidity"),
            "pressure": weather_json.get("main", {}).get("pressure"),
            "wind_speed": weather_json.get("wind", {}).get("speed"),
        })
    else:
        row.update({"temperature": None, "humidity": None, "pressure": None, "wind_speed": None})

    return row

def build_feature_row(pollution_json: dict, weather_json: Optional[dict] = None, previous_aqi: Optional[float] = None) -> dict:
    entry = pollution_json["list"][0]
    return _row_from_entry(entry, weather_json, previous_aqi)

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
    pollutant_sum DOUBLE PRECISION,
    pm_ratio DOUBLE PRECISION,
    pm25_pm10_sum DOUBLE PRECISION,
    no2_o3_ratio DOUBLE PRECISION,
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
        raise RuntimeError("SUPABASE_DB_URL is not set.")
    return create_engine(SUPABASE_DB_URL)

def drop_table(engine) -> None:
    """Nuke the table to start completely fresh."""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS aqi_features CASCADE;"))
    log.info("Dropped existing aqi_features table.")

def ensure_table(engine) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

def get_previous_aqi(engine=None) -> Optional[float]:
    if engine is None: return None
    try:
        from sqlalchemy import text
        ensure_table(engine)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT aqi FROM aqi_features ORDER BY unix_time DESC LIMIT 1")).fetchone()
        return float(result[0]) if result else None
    except Exception as e:
        return None

def write_to_feature_store(row: dict) -> None:
    write_rows_to_feature_store([row])

def write_rows_to_feature_store(rows: list, batch_size: int = 100) -> None:
    if not rows: 
        return
    
    engine = get_db_engine()
    ensure_table(engine)

    df = pd.DataFrame(rows)[DB_FEATURE_COLUMNS]
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

def run(dry_run: bool = False) -> dict:
    log.info("Fetching current air pollution + weather for %s", CITY.name)
    pollution_json = fetch_air_pollution(CITY.lat, CITY.lon)
    weather_json = fetch_weather(CITY.lat, CITY.lon)
    engine = get_db_engine() if not dry_run else None
    previous_aqi = get_previous_aqi(engine)
    row = build_feature_row(pollution_json, weather_json, previous_aqi)
    log.info("Computed feature row: AQI=%.1f, PM2.5=%.1f", row["aqi"], row["pm2_5"])
    if not dry_run: write_to_feature_store(row)
    return row

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)