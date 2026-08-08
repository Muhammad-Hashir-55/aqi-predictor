"""
Streamlit Dashboard for the Pearls AQI Predictor.
Loads the latest model from Hugging Face and the latest feature row from Supabase.
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

st.set_page_config(page_title=f"{CITY.name} AQI Predictor", page_icon="🌬️", layout="centered")

@st.cache_resource(ttl=3600)  # Cache the model for 1 hour to prevent constant re-downloading
def load_model_artifacts():
    st.info("Downloading latest model artifacts from Hugging Face Registry...")
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/random_forest/model.joblib", repo_type="model")
        scaler_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="models/random_forest/scaler.joblib", repo_type="model")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face: {e}")
        st.stop()

def get_latest_features():
    """Fetch the single most recent row from the Supabase feature store."""
    engine = create_engine(SUPABASE_DB_URL)
    query = text("SELECT * FROM aqi_features ORDER BY timestamp DESC LIMIT 1")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def main():
    st.title(f"🌬️ {CITY.name} Air Quality Forecast")
    st.write("End-to-end serverless ML pipeline predicting AQI 3 days ahead.")

    # 1. Load Model & Scaler
    model, scaler = load_model_artifacts()

    # 2. Fetch Live Data
    with st.spinner("Fetching latest feature data from Supabase..."):
        latest_data = get_latest_features()
    
    if latest_data.empty:
        st.error("No data found in the feature store.")
        return

    current_aqi = latest_data['aqi'].iloc[0]
    current_time = latest_data['timestamp'].iloc[0]
    current_hazard = classify_aqi(current_aqi)

    # 3. Predict 72 Hours Ahead
    X_input = latest_data[TRAINING_FEATURE_COLS]
    X_scaled = scaler.transform(X_input)
    predicted_aqi = model.predict(X_scaled)[0]
    predicted_hazard = classify_aqi(predicted_aqi)

    # --- Dashboard UI ---
    st.markdown("### Current Conditions")
    st.caption(f"Last updated: {current_time} UTC")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current AQI", value=f"{current_aqi:.1f}")
    with col2:
        st.metric(label="Hazard Level", value=current_hazard)

    st.divider()

    st.markdown("### 🔮 72-Hour Forecast")
    
    # Hazard Alert Logic
    if predicted_aqi > 150:
        st.error(f"⚠️ **HAZARDOUS AIR QUALITY ALERT** ⚠️\n\nForecasted AQI: **{predicted_aqi:.1f}** ({predicted_hazard})")
    elif predicted_aqi > 100:
        st.warning(f"⚠️ **POOR AIR QUALITY WARNING**\n\nForecasted AQI: **{predicted_aqi:.1f}** ({predicted_hazard})")
    else:
        st.success(f"✅ **GOOD AIR QUALITY EXPECTED**\n\nForecasted AQI: **{predicted_aqi:.1f}** ({predicted_hazard})")

    # Display underlying features for transparency
    with st.expander("View Raw Feature Data (Model Input)"):
        st.dataframe(latest_data[TRAINING_FEATURE_COLS].T, use_container_width=True)

if __name__ == "__main__":
    main()