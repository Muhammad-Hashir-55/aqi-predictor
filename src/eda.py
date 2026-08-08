"""
Exploratory Data Analysis (EDA) for the Pearls AQI Predictor.
Generates visualizations for AQI trends, pollutant correlations, and hazard distributions.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import text

from config import DB_FEATURE_COLUMNS, CITY
from feature_pipeline import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EDA_DIR = Path(__file__).parent.parent / "eda_outputs"

def fetch_data() -> pd.DataFrame:
    """Fetch the full dataset from Supabase."""
    engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM aqi_features ORDER BY timestamp"), conn)
    
    # Ensure timestamp is datetime for plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_aqi_timeseries(df: pd.DataFrame):
    """Plot the AQI trend over the last 90 days."""
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='timestamp', y='aqi', color='crimson', linewidth=1.5)
    plt.title(f"90-Day AQI Trend in {CITY.name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Air Quality Index (AQI)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(100, color='orange', linestyle='--', label='Unhealthy for Sensitive Groups (>100)')
    plt.axhline(150, color='red', linestyle='--', label='Unhealthy (>150)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "aqi_timeseries.png", dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap showing how weather impacts pollutants."""
    # Select continuous numerical columns of interest
    cols = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'temperature', 'humidity', 'wind_speed', 'pressure']
    
    # Filter only columns that actually exist in the dataframe to prevent errors
    existing_cols = [c for c in cols if c in df.columns]
    
    corr = df[existing_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix: Weather vs. Pollutants", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_matrix.png", dpi=300)
    plt.close()

def run_eda():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Fetching data from Supabase for EDA...")
    df = fetch_data()
    
    if df.empty:
        log.error("Database is empty. Run backfill_historical.py first.")
        return

    log.info(f"Loaded {len(df)} rows. Generating visualizations...")
    
    plot_aqi_timeseries(df)
    log.info("Saved time-series plot.")
    
    plot_correlation_matrix(df)
    log.info("Saved correlation matrix.")
    
    log.info(f"EDA complete! Check the '{EDA_DIR.name}' folder for the output images.")

if __name__ == "__main__":
    run_eda()