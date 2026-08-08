"""
Streamlit Dashboard for the Pearls AQI Predictor.
Loads the latest model from Hugging Face and recent feature rows from Supabase.
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from sqlalchemy import create_engine, text

# Add the src directory to the path so we can import our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import CITY, HF_MODEL_REPO, SUPABASE_DB_URL, TRAINING_FEATURE_COLS, classify_aqi

# Page Configuration
st.set_page_config(
    page_title=f"{CITY.name} AQI Predictor", 
    page_icon="🌬️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Modern Cards & Badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(ttl=3600)  # Cache model artifacts for 1 hour
def load_model_artifacts():
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/random_forest/model.joblib", repo_type="model")
        scaler_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/random_forest/scaler.joblib", repo_type="model")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model artifacts from Hugging Face: {e}")
        st.stop()

@st.cache_data(ttl=300)  # Cache feature data for 5 minutes
def get_feature_data():
    """Fetch the latest rows from the Supabase feature store for trends and prediction."""
    engine = create_engine(SUPABASE_DB_URL)
    query = text("SELECT * FROM aqi_features ORDER BY timestamp DESC LIMIT 48")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_epa_color(hazard: str) -> str:
    colors = {
        "Good": "#22C55E",
        "Moderate": "#EAB308",
        "Unhealthy for Sensitive Groups": "#F97316",
        "Unhealthy": "#EF4444",
        "Very Unhealthy": "#A855F7",
        "Hazardous": "#7E22CE"
    }
    return colors.get(hazard, "#64748B")

def main():
    # Header Section
    st.markdown(f'<p class="main-header">🌬️ {CITY.name} Air Quality Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">End-to-end 100% serverless MLOps pipeline forecasting AQI 72 hours ahead.</p>', unsafe_allow_html=True)

    # Load Model & Data
    with st.spinner("Connecting to Hugging Face Registry & Supabase Feature Store..."):
        model, scaler = load_model_artifacts()
        df_features = get_feature_data()

    if df_features.empty:
        st.error("No data found in the feature store. Please check your pipeline backfill.")
        return

    # Latest record (Current Condition)
    latest_row = df_features.iloc[0]
    current_aqi = latest_row['aqi']
    current_time = latest_row['timestamp']
    current_hazard = classify_aqi(current_aqi)

    # Prepare input for 72-hour prediction
    X_input = pd.DataFrame([latest_row[TRAINING_FEATURE_COLS]])
    X_scaled = scaler.transform(X_input)
    predicted_aqi = model.predict(X_scaled)[0]
    predicted_hazard = classify_aqi(predicted_aqi)

    # --- SIDEBAR INFO ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/airflow.png", width=64)
        st.subheader("System Status")
        st.success("🟢 Pipeline: Active (Hourly)")
        st.info("☁️ Registry: Hugging Face Hub")
        st.info("🗄️ Store: Supabase PostgreSQL")
        st.markdown("---")
        st.markdown("**Author:** Muhammad Hashir Awaiz")
        st.markdown("**Institution:** GIKI")

    # --- CURRENT CONDITIONS METRICS ---
    st.markdown("### 📍 Current Live Conditions")
    st.caption(f"Last synchronized timestamp: {current_time} UTC")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Current AQI", value=f"{current_aqi:.1f}", delta=f"{latest_row.get('aqi_change_rate', 0):+.1f} vs last hr")
    with col2:
        st.metric(label="Hazard Status", value=current_hazard)
    with col3:
        st.metric(label="PM2.5 Concentration", value=f"{latest_row.get('pm2_5', 0):.1f} µg/m³")
    with col4:
        st.metric(label="Wind Speed", value=f"{latest_row.get('wind_speed', 0):.1f} m/s")

    st.markdown("")

    # --- 72-HOUR FORECAST SECTION ---
    st.markdown("### 🔮 72-Hour Ahead Predictive Intelligence")
    
    forecast_col1, forecast_col2 = st.columns([2, 1])
    
    with forecast_col1:
        if predicted_aqi > 150:
            st.error(f"⚠️ **SEVERE HAZARD ALERT**\n\nForecasted AQI in 72 Hours: **{predicted_aqi:.1f}** ({predicted_hazard})\n\n*Advisory:* Avoid all outdoor physical activities.")
        elif predicted_aqi > 100:
            st.warning(f"⚠️ **POOR AIR QUALITY WARNING**\n\nForecasted AQI in 72 Hours: **{predicted_aqi:.1f}** ({predicted_hazard})\n\n*Advisory:* Sensitive groups should limit prolonged outdoor exertion.")
        else:
            st.success(f"✅ **FAVORABLE AIR QUALITY**\n\nForecasted AQI in 72 Hours: **{predicted_aqi:.1f}** ({predicted_hazard})\n\n*Advisory:* Air quality is acceptable for outdoor activities.")

    with forecast_col2:
        st.metric(label="Predicted 3-Day AQI", value=f"{predicted_aqi:.1f}", delta=f"{predicted_aqi - current_aqi:+.1f} shift")

    st.markdown("---")

    # --- RECENT TRENDS CHART ---
    st.markdown("### 📈 Recent 48-Hour AQI Trend & Smoothing")
    chart_df = df_features[['timestamp', 'aqi', 'aqi_rolling_mean_24h']].sort_values('timestamp')
    chart_df = chart_df.set_index('timestamp')
    st.line_chart(chart_df, color=["#EF4444", "#1E3A8Y"])

    # --- TRANSPARENCY & DEBUG EXPANDER ---
    with st.expander("🔍 View Raw 37-Feature Model Input Vector"):
        st.dataframe(latest_row[TRAINING_FEATURE_COLS].to_frame(name="Feature Value"), use_container_width=True)

if __name__ == "__main__":
    main()