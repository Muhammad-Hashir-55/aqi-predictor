# Pearls AQI Predictor

End-to-end, 100% serverless ML pipeline that forecasts the Air Quality Index (AQI)
for **Sialkot, Pakistan** three days ahead — feature pipeline, training pipeline,
automated CI/CD, and an interactive dashboard.

## Tech Stack

- **Language:** Python 3.12
- **Data source:** OpenWeather (Air Pollution + Weather APIs)
- **Feature Store / Model Registry:** Hopsworks (free tier)
- **Modeling:** scikit-learn (Random Forest, Ridge Regression), PyTorch (deep learning)
- **Explainability:** SHAP
- **Automation:** GitHub Actions
- **Dashboard:** Streamlit / FastAPI

## Project Structure

\```
aqi-predictor/
├── .env.example              # template for required API keys
├── requirements.txt
├── src/
│   ├── config.py              # city coords, secrets, AQI hazard classifier
│   ├── feature_pipeline.py    # fetch -> compute features -> write to Hopsworks
│   ├── backfill_historical.py     [next]
│   └── training_pipeline.py       [next]
├── app/
│   └── dashboard.py                [next] Streamlit dashboard
├── .github/workflows/
│   ├── feature_pipeline.yml        [next] hourly cron
│   └── training_pipeline.yml       [next] daily cron
├── tests/
└── data/                       # local scratch space, gitignored
\```

## Setup

\```bash
git clone <repo-url>
cd aqi-predictor
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env            # then fill in your real keys
\```

Required keys in `.env`:
- `OPENWEATHER_API_KEY` — from [home.openweathermap.org/api_keys](https://home.openweathermap.org/api_keys)
- `HOPSWORKS_API_KEY` — from your [app.hopsworks.ai](https://app.hopsworks.ai) project's Account Settings
- `HOPSWORKS_PROJECT` — defaults to `aqi_predictor`

> **Windows note:** installing Hopsworks may require the
> [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (Desktop development with C++ workload) to compile one of its dependencies (`twofish`).

## Running

\```bash
# Fetch + compute features without writing to Hopsworks (sanity check)
python src/feature_pipeline.py --dry-run

# Full run — writes to the Hopsworks feature store
python src/feature_pipeline.py
\```

## Progress So Far

- [x] Project scaffolded, Python 3.12 environment working end to end on Windows
- [x] `config.py` — city config, secrets loading, AQI hazard-level classifier
- [x] `feature_pipeline.py` — fetches OpenWeather air pollution + weather data,
      converts PM2.5 to a 0-500 EPA AQI value, computes time-based features
      (hour/day/month/weekend) and derived features (AQI change rate)
- [x] Logic unit-tested against mocked API payloads
- [ ] API keys being wired in — pending a live dry-run test
- [ ] Historical backfill script
- [ ] Training pipeline (Random Forest, Ridge, PyTorch models)
- [ ] GitHub Actions automation (hourly feature pipeline, daily training)
- [ ] Streamlit dashboard with SHAP explanations + hazard alerts
- [ ] Final report

## License

Academic project — no license specified yet.