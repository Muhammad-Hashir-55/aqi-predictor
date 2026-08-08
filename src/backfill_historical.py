"""
Historical backfill for the Pearls AQI Predictor with 37-feature sequencing.
Merges OpenWeather pollution data with actual historical weather data from Open-Meteo.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from config import CITY
from feature_pipeline import (
    fetch_air_pollution_range,
    pm25_to_aqi,
    engineer_features_dataframe,
    get_db_engine,
    drop_table,
    ensure_table,
    get_previous_aqi,
    write_rows_to_feature_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHUNK_HOURS = 24 * 7  # one API call per 7-day window

def _chunk_ranges(start: datetime, end: datetime, chunk_hours: int = CHUNK_HOURS):
    """Yield time ranges in chunks for batch processing."""
    cursor = start
    step = timedelta(hours=chunk_hours)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end

def fetch_weather_chunk(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> dict:
    """
    Fetch real historical weather from Open-Meteo and index it by exact UNIX timestamp.
    Includes wind direction for advanced 37-feature engineering.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    weather_lookup = {}
    if "hourly" in data:
        hourly = data["hourly"]
        for i, time_str in enumerate(hourly["time"]):
            # Open-Meteo returns ISO strings like '2026-05-10T11:00'
            dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
            ts = int(dt.timestamp())
            
            # Pack it into the shape _row_from_entry expects
            weather_lookup[ts] = {
                "main": {
                    "temp": hourly["temperature_2m"][i],
                    "humidity": hourly["relative_humidity_2m"][i],
                    "pressure": hourly["surface_pressure"][i],
                },
                "wind": {
                    "speed": hourly["wind_speed_10m"][i],
                    "deg": hourly["wind_direction_10m"][i] if "wind_direction_10m" in hourly else 0.0
                }
            }
    return weather_lookup

def fetch_openmeteo_weather_chunk(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> dict:
    """
    Alternative format for weather data retrieval (flatter structure).
    Used for compatibility with newer feature engineering approach.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    lookup = {}
    if "hourly" in data:
        hourly = data["hourly"]
        for i, time_str in enumerate(hourly["time"]):
            dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
            ts = int(dt.timestamp())
            lookup[ts] = {
                "temp": hourly["temperature_2m"][i],
                "humidity": hourly["relative_humidity_2m"][i],
                "pressure": hourly["surface_pressure"][i],
                "wind_speed": hourly["wind_speed_10m"][i],
                "wind_direction": hourly["wind_direction_10m"][i] if "wind_direction_10m" in hourly else 0.0,
            }
    return lookup

def _build_raw_row(entry: dict, weather_data: dict) -> dict:
    """
    Build a raw feature row from pollution and weather data.
    Returns a row with basic fields (no engineered features).
    """
    components = entry["components"]
    ts_unix = entry["dt"]
    pm25 = components.get("pm2_5", 0.0)
    
    return {
        "city": CITY.name,
        "timestamp": datetime.fromtimestamp(ts_unix, tz=timezone.utc),
        "unix_time": ts_unix,
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": pm25,
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
        "aqi": pm25_to_aqi(pm25),
        "temperature": weather_data.get("temp"),
        "humidity": weather_data.get("humidity"),
        "pressure": weather_data.get("pressure"),
        "wind_speed": weather_data.get("wind_speed"),
        "wind_direction": weather_data.get("wind_direction", 0.0),
    }

def run_backfill(days_back: int = 90, dry_run: bool = False, request_delay: float = 1.0) -> list:
    """
    Backfill historical data with full 37-feature engineering.
    
    Args:
        days_back: Number of days to backfill.
        dry_run: If True, don't write to database.
        request_delay: Delay between API requests (seconds).
    
    Returns:
        List of fully engineered feature rows.
    """
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)

    engine = None
    previous_aqi = None
    if not dry_run:
        engine = get_db_engine()
        drop_table(engine)  # Start completely fresh for consistent backfill
        ensure_table(engine)
        previous_aqi = get_previous_aqi(engine)

    ranges = list(_chunk_ranges(start, end))
    log.info("Backfilling %s from %s to %s across %d chunks", CITY.name, start, end, len(ranges))

    raw_rows = []
    for i, (chunk_start, chunk_end) in enumerate(ranges, 1):
        try:
            # Fetch pollution and weather for this specific time chunk
            pollution_payload = fetch_air_pollution_range(CITY.lat, CITY.lon, chunk_start, chunk_end)
            
            # Use both weather formats for compatibility
            # The flat format is used for building raw rows
            weather_lookup_flat = fetch_openmeteo_weather_chunk(CITY.lat, CITY.lon, chunk_start, chunk_end)
            # The nested format is used for legacy compatibility
            weather_lookup_nested = fetch_weather_chunk(CITY.lat, CITY.lon, chunk_start, chunk_end)
            
            entries = pollution_payload.get("list", [])
            for entry in entries:
                ts_unix = entry["dt"]
                w_data = weather_lookup_flat.get(ts_unix, {})
                
                # Build raw row (without engineered features)
                raw_row = _build_raw_row(entry, w_data)
                
                # For backward compatibility, also track previous AQI if needed
                if previous_aqi is not None:
                    raw_row["aqi_change_rate"] = raw_row["aqi"] - previous_aqi
                previous_aqi = raw_row["aqi"]
                
                raw_rows.append(raw_row)
                
            log.info("Chunk %d/%d (%s to %s): matched %d hourly rows", 
                    i, len(ranges), chunk_start.date(), chunk_end.date(), len(entries))
        except Exception as e:
            log.warning("Chunk %d/%d failed, skipping: %s", i, len(ranges), e)

        time.sleep(request_delay)

    if not raw_rows:
        log.error("No historical rows fetched. Aborting backfill.")
        return []

    log.info("Raw backfill complete: %d rows fetched", len(raw_rows))
    
    # Convert to DataFrame and engineer all 37 features
    df = pd.DataFrame(raw_rows)
    log.info("Engineering 37 features across %d historical rows...", len(df))
    engineered_df = engineer_features_dataframe(df)
    
    # Drop initial rows where 24h lags are NaN to maintain clean training input
    # This ensures we have valid lag features for all training samples
    before_drop = len(engineered_df)
    engineered_df = engineered_df.dropna(subset=["aqi_lag_24h"]).reset_index(drop=True)
    dropped = before_drop - len(engineered_df)
    if dropped > 0:
        log.info("Dropped %d initial rows with NaN lag features", dropped)
    
    records = engineered_df.to_dict(orient="records")
    log.info("Backfill sequence complete: %d clean records ready for training", len(records))

    if not dry_run:
        write_rows_to_feature_store(records)
    else:
        log.info("Dry run — skipping feature store write.")

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, 
                       help="Number of days to backfill (default: 90)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without writing to database")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API requests in seconds (default: 1.0)")
    args = parser.parse_args()
    run_backfill(days_back=args.days, dry_run=args.dry_run, request_delay=args.delay)