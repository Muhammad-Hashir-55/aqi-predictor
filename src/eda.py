"""
Exploratory Data Analysis (EDA) for the Pearls AQI Predictor.
Generates advanced visualizations for 37-feature metrics, thermal inversions, and rolling trends.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import text

from config import CITY
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

def plot_aqi_timeseries_with_rolling(df: pd.DataFrame):
    """Plot AQI trend alongside its 24-hour rolling mean for trend smoothing."""
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='timestamp', y='aqi', color='crimson', alpha=0.35, label='Raw AQI')
    if 'aqi_rolling_mean_24h' in df.columns:
        sns.lineplot(data=df, x='timestamp', y='aqi_rolling_mean_24h', color='darkblue', linewidth=2, label='24h Rolling Mean')
    
    plt.title(f"90-Day AQI Trend & Smoothing in {CITY.name}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Air Quality Index (AQI)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(100, color='orange', linestyle='--', label='Unhealthy for Sensitive Groups (>100)')
    plt.axhline(150, color='red', linestyle='--', label='Unhealthy (>150)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "aqi_timeseries_smoothed.png", dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap showing correlations across key raw and engineered 37 features."""
    cols = [
        'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'temperature', 'humidity', 
        'wind_speed', 'pressure', 'aqi_lag_24h', 'pm25_rolling_mean_6h', 
        'thermal_inversion_flag', 'temp_humidity_index', 'wind_u', 'wind_v'
    ]
    
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix: Engineered Physics & Lags vs. AQI", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_matrix_37_features.png", dpi=300)
    plt.close()

def plot_thermal_inversion_impact(df: pd.DataFrame):
    """Boxplot illustrating AQI distribution during thermal inversion events."""
    if 'thermal_inversion_flag' not in df.columns:
        return
        
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='thermal_inversion_flag', y='aqi', palette=['skyblue', 'salmon'])
    plt.title("Impact of Thermal Inversion on AQI Levels", fontsize=16, fontweight='bold')
    plt.xlabel("Thermal Inversion Flag (0 = Normal, 1 = Trapped Pollutants)", fontsize=12)
    plt.ylabel("Air Quality Index (AQI)", fontsize=12)
    plt.xticks([0, 1], ['Normal Conditions', 'Thermal Inversion'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "thermal_inversion_impact.png", dpi=300)
    plt.close()

def run_eda():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Fetching data from Supabase for EDA...")
    df = fetch_data()
    
    if df.empty:
        log.error("Database is empty. Run backfill_historical.py first.")
        return

    log.info(f"Loaded {len(df)} rows. Generating 37-feature visualizations...")
    
    plot_aqi_timeseries_with_rolling(df)
    log.info("Saved smoothed time-series plot.")
    
    plot_correlation_matrix(df)
    log.info("Saved extended correlation matrix.")

    plot_thermal_inversion_impact(df)
    log.info("Saved thermal inversion impact plot.")
    
    log.info(f"EDA complete! Check the '{EDA_DIR.name}' folder for the output images.")

if __name__ == "__main__":
    run_eda()