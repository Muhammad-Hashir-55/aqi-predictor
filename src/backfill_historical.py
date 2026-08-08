"""
Historical backfill for the Pearls AQI Predictor.
Now merges OpenWeather pollution data with actual historical weather data from Open-Meteo.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from config import CITY
from feature_pipeline import (
    fetch_air_pollution_range,
    _row_from_entry,
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
    cursor = start
    step = timedelta(hours=chunk_hours)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end

def fetch_weather_chunk(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> dict:
    """Fetch real historical weather from Open-Meteo and index it by exact UNIX timestamp."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
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
                    "speed": hourly["wind_speed_10m"][i]
                }
            }
    return weather_lookup

def run_backfill(days_back: int = 30, dry_run: bool = False, request_delay: float = 1.0) -> list:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)

    engine = None
    previous_aqi = None
    if not dry_run:
        engine = get_db_engine()
        drop_table(engine)  # Start completely fresh
        ensure_table(engine)
        previous_aqi = get_previous_aqi(engine)

    ranges = list(_chunk_ranges(start, end))
    log.info("Backfilling %s from %s to %s across %d chunks", CITY.name, start, end, len(ranges))

    all_rows = []
    for i, (chunk_start, chunk_end) in enumerate(ranges, 1):
        try:
            # Fetch pollution and weather for this specific time chunk
            pollution_payload = fetch_air_pollution_range(CITY.lat, CITY.lon, chunk_start, chunk_end)
            weather_lookup = fetch_weather_chunk(CITY.lat, CITY.lon, chunk_start, chunk_end)
            
            entries = pollution_payload.get("list", [])
            for entry in entries:
                # Marry the pollution entry with the matching weather timestamp
                w_json = weather_lookup.get(entry["dt"])
                row = _row_from_entry(entry, weather_json=w_json, previous_aqi=previous_aqi)
                previous_aqi = row["aqi"]
                all_rows.append(row)
                
            log.info("Chunk %d/%d (%s to %s): matched %d hourly rows", i, len(ranges), chunk_start.date(), chunk_end.date(), len(entries))
        except Exception as e:
            log.warning("Chunk %d/%d failed, skipping: %s", i, len(ranges), e)

        time.sleep(request_delay)

    log.info("Backfill complete: %d total rows computed", len(all_rows))

    if dry_run:
        log.info("Dry run — skipping feature store write.")
    else:
        write_rows_to_feature_store(all_rows)

    return all_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    run_backfill(days_back=args.days, dry_run=args.dry_run, request_delay=args.delay)