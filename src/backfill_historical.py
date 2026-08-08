"""
Historical backfill for the Pearls AQI Predictor.

Runs the feature pipeline's fetch/compute logic over a range of past dates,
so training_pipeline.py has actual historical data to train on.

Fetches in day-sized chunks (not hour-by-hour) to stay well within OpenWeather's
free-tier rate limit — e.g. 30 days needs ~5 API calls instead of 720.

Note: OpenWeather's free tier has no historical *weather* endpoint (only current),
so backfilled rows carry pollution + time features but temperature/humidity/
pressure/wind_speed are null. This matches what build_feature_row() already
does when weather_json is None — the training pipeline handles nulls accordingly.

Run:
    python src/backfill_historical.py --days 30
    python src/backfill_historical.py --days 30 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from config import CITY
from feature_pipeline import (
    fetch_air_pollution_range,
    _row_from_entry,
    get_db_engine,
    ensure_table,
    get_previous_aqi,
    write_rows_to_feature_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHUNK_HOURS = 24 * 7  # one API call per 7-day window


def _chunk_ranges(start: datetime, end: datetime, chunk_hours: int = CHUNK_HOURS):
    """Yield (chunk_start, chunk_end) pairs covering [start, end]."""
    cursor = start
    step = timedelta(hours=chunk_hours)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


def run_backfill(days_back: int = 30, dry_run: bool = False, request_delay: float = 1.0) -> list:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)

    engine = None
    previous_aqi = None
    if not dry_run:
        engine = get_db_engine()
        ensure_table(engine)
        previous_aqi = get_previous_aqi(engine)

    ranges = list(_chunk_ranges(start, end))
    log.info("Backfilling %s from %s to %s across %d API call(s)", CITY.name, start, end, len(ranges))

    all_rows = []
    for i, (chunk_start, chunk_end) in enumerate(ranges, 1):
        try:
            payload = fetch_air_pollution_range(CITY.lat, CITY.lon, chunk_start, chunk_end)
            entries = payload.get("list", [])
            for entry in entries:
                row = _row_from_entry(entry, weather_json=None, previous_aqi=previous_aqi)
                previous_aqi = row["aqi"]
                all_rows.append(row)
            log.info("Chunk %d/%d (%s to %s): got %d hourly rows", i, len(ranges), chunk_start.date(), chunk_end.date(), len(entries))
        except Exception as e:
            log.warning("Chunk %d/%d failed, skipping: %s", i, len(ranges), e)

        time.sleep(request_delay)  # be polite to the free-tier rate limit

    log.info("Backfill complete: %d total rows fetched", len(all_rows))

    if dry_run:
        log.info("Dry run — skipping feature store write.")
    else:
        write_rows_to_feature_store(all_rows)

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI features")
    parser.add_argument("--days", type=int, default=30, help="How many days back to backfill (default 30)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + compute only, skip the feature store write")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between API calls (default 1.0)")
    args = parser.parse_args()
    run_backfill(days_back=args.days, dry_run=args.dry_run, request_delay=args.delay)