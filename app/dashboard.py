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

# Enhanced Custom CSS Styling
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary: #3B82F6;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #6366F1;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px solid #E2E8F0;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #D1FAE5;
        border-left-color: #10B981;
        color: #065F46;
    }
    .alert-warning {
        background-color: #FEF3C7;
        border-left-color: #F59E0B;
        color: #92400E;
    }
    .alert-danger {
        background-color: #FEE2E2;
        border-left-color: #EF4444;
        color: #991B1B;
    }
    .stAlert {
        border-radius: 10px;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #E2E8F0;
    }
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
        st.error(f"❌ Failed to load model artifacts from Hugging Face: {e}")
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
    # Header Section with gradient
    st.markdown(f'<p class="main-header">🌬️ {CITY.name} Air Quality Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ End-to-end serverless MLOps pipeline forecasting AQI 72 hours ahead with Random Forest</p>', unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Load Model & Data
    with st.spinner("🔄 Connecting to Hugging Face Registry & Supabase Feature Store..."):
        model, scaler = load_model_artifacts()
        df_features = get_feature_data()

    if df_features.empty:
        st.error("⚠️ No data found in the feature store. Please check your pipeline backfill.")
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
        st.markdown("""
        <a href="https://sialkot-aqi-predictor.streamlit.app" target="_blank">
            <img src="https://img.icons8.com/color/96/airflow.png" width="80" alt="AQI Predictor">
        </a>
        """, unsafe_allow_html=True)
        st.markdown("## 🎛️ System Status")
        st.success("✅ **Pipeline:** Active (Hourly)")
        st.info("☁️ **Registry:** Hugging Face Hub")
        st.info("🗄️ **Store:** Supabase PostgreSQL")
        st.markdown("---")
        st.markdown("### 👤 About")
        st.markdown("**Author:** Muhammad Hashir Awaiz")
        st.markdown("**Institution:** GIKI")
        st.markdown("**Program:** BS Artificial Intelligence")
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[📊 Live Dashboard](https://sialkot-aqi-predictor.streamlit.app)")
        st.markdown("[💻 GitHub Repository](https://github.com/Muhammad-Hashir-55/aqi-predictor)")
        st.markdown("[🤖 Model Registry](https://huggingface.co/HashirAwaiz/aqi-forecast-model)")

    # --- CURRENT CONDITIONS METRICS ---
    st.markdown('<p class="section-header">📍 Current Live Conditions</p>', unsafe_allow_html=True)
    st.caption(f"🕐 Last synchronized: {current_time} UTC")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌡️ Current AQI",
            value=f"{current_aqi:.1f}",
            delta=f"{latest_row.get('aqi_change_rate', 0):+.1f} vs last hr",
            help="Current Air Quality Index"
        )
    with col2:
        hazard_color = get_epa_color(current_hazard)
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <p style='font-size: 0.9rem; color: #64748B; margin-bottom: 5px;'>⚠️ Hazard Status</p>
            <p style='font-size: 1.5rem; font-weight: bold; color: {hazard_color}; margin: 0;'>{current_hazard}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.metric(
            label="💨 PM2.5 Concentration",
            value=f"{latest_row.get('pm2_5', 0):.1f} µg/m³",
            help="Fine particulate matter concentration"
        )
    with col4:
        st.metric(
            label="🌬️ Wind Speed",
            value=f"{latest_row.get('wind_speed', 0):.1f} m/s",
            help="Current wind speed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 72-HOUR FORECAST SECTION ---
    st.markdown('<p class="section-header">🔮 72-Hour Ahead Predictive Intelligence</p>', unsafe_allow_html=True)
    st.caption("🤖 Powered by Random Forest model trained on 42 engineered features")
    
    forecast_col1, forecast_col2 = st.columns([3, 1])
    
    with forecast_col1:
        if predicted_aqi > 150:
            st.markdown(f"""
            <div class='alert-box alert-danger'>
                <h3 style='margin-top: 0;'>⚠️ SEVERE HAZARD ALERT</h3>
                <p style='font-size: 1.2rem; margin: 10px 0;'>
                    <strong>Forecasted AQI in 72 Hours:</strong> 
                    <span style='font-size: 1.5rem; font-weight: bold;'>{predicted_aqi:.1f}</span> 
                    ({predicted_hazard})
                </p>
                <p style='margin: 10px 0;'><strong>🚨 Advisory:</strong> Avoid all outdoor physical activities. Stay indoors with air filtration if possible.</p>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_aqi > 100:
            st.markdown(f"""
            <div class='alert-box alert-warning'>
                <h3 style='margin-top: 0;'>⚠️ POOR AIR QUALITY WARNING</h3>
                <p style='font-size: 1.2rem; margin: 10px 0;'>
                    <strong>Forecasted AQI in 72 Hours:</strong> 
                    <span style='font-size: 1.5rem; font-weight: bold;'>{predicted_aqi:.1f}</span> 
                    ({predicted_hazard})
                </p>
                <p style='margin: 10px 0;'><strong>⚠️ Advisory:</strong> Sensitive groups should limit prolonged outdoor exertion.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-box alert-success'>
                <h3 style='margin-top: 0;'>✅ FAVORABLE AIR QUALITY</h3>
                <p style='font-size: 1.2rem; margin: 10px 0;'>
                    <strong>Forecasted AQI in 72 Hours:</strong> 
                    <span style='font-size: 1.5rem; font-weight: bold;'>{predicted_aqi:.1f}</span> 
                    ({predicted_hazard})
                </p>
                <p style='margin: 10px 0;'><strong>✅ Advisory:</strong> Air quality is acceptable for outdoor activities.</p>
            </div>
            """, unsafe_allow_html=True)

    with forecast_col2:
        delta_color = "normal" if (predicted_aqi - current_aqi) <= 0 else "inverse"
        st.metric(
            label="🎯 Predicted 3-Day AQI",
            value=f"{predicted_aqi:.1f}",
            delta=f"{predicted_aqi - current_aqi:+.1f} shift",
            help="72-hour ahead forecast"
        )
        
        # Add a visual indicator
        if predicted_aqi > current_aqi:
            st.markdown(f"<p style='text-align: center; color: #EF4444; font-size: 0.9rem;'>📈 AQI expected to worsen</p>", unsafe_allow_html=True)
        elif predicted_aqi < current_aqi:
            st.markdown(f"<p style='text-align: center; color: #10B981; font-size: 0.9rem;'>📉 AQI expected to improve</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; color: #64748B; font-size: 0.9rem;'>➡️ AQI expected to remain stable</p>", unsafe_allow_html=True)

    st.markdown("---")

    # --- RECENT TRENDS CHART ---
    st.markdown('<p class="section-header">📈 Recent 48-Hour AQI Trend & Smoothing</p>', unsafe_allow_html=True)
    st.caption("📊 Real-time AQI values with 24-hour rolling average")
    
    chart_df = df_features[['timestamp', 'aqi', 'aqi_rolling_mean_24h']].sort_values('timestamp')
    chart_df = chart_df.set_index('timestamp')
    st.line_chart(chart_df, color=["#EF4444", "#3B82F6"])

    # --- MODEL PERFORMANCE SECTION ---
    st.markdown('<p class="section-header">🎯 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric(
            label="📉 RMSE (Root Mean Squared Error)",
            value="33.34",
            help="Lower is better. Random Forest achieved 34.7% improvement over baseline."
        )
    with perf_col2:
        st.metric(
            label="📊 MAE (Mean Absolute Error)",
            value="23.02",
            help="Average prediction error in AQI units. 30.2% better than baseline."
        )
    with perf_col3:
        st.metric(
            label="💯 R² Score",
            value="0.71",
            help="Explains 71% of variance in AQI. Positive and strong performance."
        )

    st.markdown("---")

    # --- TRANSPARENCY & DEBUG EXPANDER ---
    with st.expander("🔍 View Raw 42-Feature Model Input Vector"):
        st.caption("📋 Complete feature vector used for the 72-hour prediction")
        feature_df = latest_row[TRAINING_FEATURE_COLS].to_frame(name="Feature Value")
        st.dataframe(feature_df, use_container_width=True)
        
        # Add download button
        csv = feature_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Feature Vector as CSV",
            data=csv,
            file_name='feature_vector.csv',
            mime='text/csv',
        )

    # --- FOOTER ---
    st.markdown("""
    <div class='footer'>
        <p style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>🌬️ Pearls AQI Predictor</p>
        <p style='margin: 0.5rem 0;'>End-to-End Serverless MLOps Pipeline | Built with ❤️ by Muhammad Hashir Awaiz</p>
        <p style='margin: 0.5rem 0;'>GIKI | BS Artificial Intelligence | 2024</p>
        <p style='font-size: 0.85rem; color: #94A3B8; margin-top: 1rem;'>
            Model: Random Forest | Features: 42 | Forecast Horizon: 72 hours
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()