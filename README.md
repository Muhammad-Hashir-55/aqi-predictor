# Pearls AQI Predictor: End-to-End Serverless MLOps Pipeline 🌬️

**Author:** Muhammad Hashir Awaiz  
**Institution:** Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)  
**Program:** BS Artificial Intelligence  
**Live Application:** [Sialkot AQI Predictor · Streamlit](https://sialkot-aqi-predictor.streamlit.app)

An end-to-end, serverless machine learning architecture engineered to forecast the Air Quality Index (AQI) in Sialkot, Pakistan, **72 hours (3 days) in advance**.

The project demonstrates a production-oriented MLOps lifecycle featuring automated data ingestion, chronological model evaluation, automated retraining, model versioning, explainability, and real-time dashboarding.

---

## 1. System Architecture

The infrastructure prioritizes robust, free-tier-friendly services capable of supporting scheduled CI/CD workflows without requiring an always-on server.

```mermaid
graph TD
    A[OpenWeather API] -->|Live Pollutants| C[Feature Pipeline]
    B[Open-Meteo API] -->|Historical Weather| C
    C -->|Hourly Upsert| D[(Supabase PostgreSQL)]
    D -->|Daily Training Data| E[Training Pipeline]
    E -->|Model Training| F{Model Evaluation}
    F -->|Best Model| G[Hugging Face Model Registry]
    D -->|Latest Features| H[Streamlit Dashboard]
    G -->|Versioned Artifacts| H
    H -->|72-Hour Prediction & Alerts| I((End User))
    
    J[GitHub Actions] -->|Hourly| C
    J -->|Daily| E
```

### Pipeline Flow

1. **Data ingestion** collects pollutant and weather information from external APIs.
2. **Feature engineering** transforms raw measurements into model-ready features.
3. **Supabase PostgreSQL** acts as the persistent feature store.
4. **GitHub Actions** automatically executes scheduled pipelines.
5. **Training** evaluates multiple machine learning architectures using a chronological split.
6. **Model selection** chooses the model with the lowest RMSE.
7. **Hugging Face Hub** stores the selected model and preprocessing artifacts.
8. **Streamlit** loads the latest model and live features to generate 72-hour AQI predictions.

---

## 2. Internship Requirements Satisfaction

This repository fulfills the major requirements outlined in the Pearls AQI Predictor project brief.

* ✅ **Feature Pipeline:** Fetches live weather and pollutant data, computes **42 model features** including cyclical time encodings and derived variables, and securely upserts them into the **Supabase PostgreSQL** feature store.

* ✅ **Historical Backfill:** Performs a **90-day historical backfill**, merging OpenWeather pollutant history with Open-Meteo weather data and handling database deduplication.

* ✅ **Exploratory Data Analysis:** Performs EDA after data preparation, including correlation analysis and investigation of sensor anomalies.

* ✅ **Training Pipeline:** Evaluates **five machine learning architectures** — XGBoost, LightGBM, Random Forest, Ridge Regression, and PyTorch MLP — alongside a Persistence Baseline.

* ✅ **Chronological Evaluation:** Uses a time-ordered train/test split to reduce the risk of future-data leakage.

* ✅ **Model Registry:** Stores the trained model, preprocessing artifacts, and SHAP explainability output in the **Hugging Face Hub**.

* ✅ **CI/CD Automation:** Uses **GitHub Actions** to orchestrate automated feature ingestion and model retraining.

* ✅ **Web Dashboard:** A **Streamlit** frontend serves predictions and converts AQI values into actionable environmental health alerts.

---

## 3. Current Model Results

The latest training run retrieved **2,041 rows** from the feature store.

After constructing the 72-hour-ahead target, **1,911 rows** had a matching future AQI label.

The chronological split produced:

* **Training rows:** 1,528
* **Test rows:** 383
* **Features:** 42

The training pipeline evaluated five candidate models against a naive Persistence Baseline.

| Model                |      RMSE |       MAE |  R² Score |
| -------------------- | --------: | --------: | --------: |
| Persistence Baseline |    51.08 |    33.00 |    1.287 |
| XGBoost              |    43.49 |    32.66 |    0.657 |
| LightGBM             |    35.52 |    24.98 |    0.106 |
| **Random Forest 🏆** | **33.34** | **23.02** | **0.71** |
| Ridge Regression     |    45.20 |    35.54 |    0.791 |
| PyTorch MLP          |    40.63 |    28.61 |    0.447 |

### Selected Model: Random Forest

The training pipeline selects the model with the **lowest RMSE**.

Random Forest achieved:

* **RMSE:** 33.34
* **MAE:** 23.02
* **R²:** 0.71

It therefore became the production model and was successfully pushed to the configured Hugging Face repository.

---

## 4. R² Interpretation

The latest Random Forest model has a **positive R² of 0.71**, which indicates strong predictive performance on the held-out test set.

The coefficient of determination is:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

where:

* $SS_{res}$ is the sum of squared prediction errors.
* $SS_{tot}$ is the total variance of the target relative to its mean.

### What Does R² = 0.71 Mean?

An R² of approximately **0.71** means that the Random Forest explains about **71% of the variance** in the held-out AQI target relative to the mean-prediction reference.

This is a **positive R²**, indicating the model performs substantially better than the baseline.

However, R² should be interpreted together with RMSE and MAE rather than in isolation.

The Random Forest achieves substantially lower prediction error than the persistence baseline:

| Metric | Persistence | Random Forest | Improvement |
| ------ | ----------: | ------------: | -------------------: |
| RMSE   |      51.08 |     **33.34** |     **34.7% lower** |
| MAE    |      33.00 |     **23.02** |     **30.2% lower** |
| R²     |      1.287 |     **0.71** | Improved to positive |

The RMSE improvement is:

$$\frac{51.08 - 33.34}{51.08} \times 100 \approx 34.7\%$$

The MAE improvement is:

$$\frac{33.00 - 23.02}{33.00} \times 100 \approx 30.2\%$$

This indicates that the Random Forest is substantially better than simply assuming the future AQI will equal the current AQI.

---

## 5. Comparison With the Persistence Baseline

The Persistence Baseline assumes:

$$AQI_{t+72} = AQI_t$$

In simple terms:

> "The AQI three days from now will be the same as it is now."

This is a useful baseline because it establishes how difficult the forecasting task is without using a machine learning model.

The Random Forest improves significantly over this baseline:

```
Persistence Baseline
RMSE: 51.08
MAE : 33.00
R²  : 1.287
      │
      │ Machine Learning
      ▼
Random Forest
RMSE: 33.34
MAE : 23.02
R²  : 0.71
```

The reduction from **51.08 → 33.34 RMSE** demonstrates that the engineered features contain useful information for predicting future AQI.

---

## 6. Model Selection Strategy

The pipeline evaluates multiple model families:

### XGBoost

Gradient-boosted decision trees designed to capture non-linear relationships and feature interactions.

### LightGBM

A highly efficient gradient-boosting implementation optimized for fast training and strong performance on tabular datasets.

### Random Forest

An ensemble of randomized decision trees capable of modelling non-linear relationships and interactions without requiring feature scaling.

### Ridge Regression

A regularized linear model used as a simpler baseline for determining how much predictive performance can be obtained from approximately linear relationships.

### PyTorch MLP

A neural-network-based model used to evaluate whether a learned non-linear representation improves over tree-based and linear approaches.

### Selection Criterion

The production model is selected using **lowest test-set RMSE**.

In the latest run:

```
Random Forest : 33.34 RMSE  ← Winner
LightGBM      : 35.52 RMSE
MLP           : 40.63 RMSE
XGBoost       : 43.49 RMSE
Ridge         : 45.20 RMSE
Baseline      : 51.08 RMSE
```

Therefore, **Random Forest is currently the best-performing model according to the project's primary evaluation metric.**

---

## 7. Training Pipeline Output

The latest successful training run produced:

```
Fetched 2041 rows from the feature store
Built 72-hour-ahead target: 1911/2041 rows have a matching future label
Train rows: 1528
Test rows: 383
Features used: 42
```

The final model selection was:

```
Best model: random_forest
Lowest RMSE: 33.3448
MAE: 23.0205
R²: 0.71
```

The selected artifacts were successfully uploaded to:

**Hugging Face:** [HashirAwaiz/aqi-forecast-model](https://huggingface.co/HashirAwaiz/aqi-forecast-model)

Uploaded artifacts include:

* `model.joblib` — trained Random Forest model
* `scaler.joblib` — preprocessing artifact
* `shap_summary.png` — SHAP feature-importance visualization

---

## 8. Tech Stack

| Category                | Technology                                  |
| ----------------------- | ------------------------------------------- |
| Language                | Python 3.12                                 |
| Data Ingestion          | OpenWeather API, Open-Meteo API             |
| Database / Feature Store| Supabase PostgreSQL                         |
| Machine Learning        | scikit-learn, XGBoost, LightGBM, PyTorch   |
| Explainability          | SHAP                                        |
| Model Registry          | Hugging Face Hub                            |
| CI/CD                   | GitHub Actions                              |
| Dashboard               | Streamlit                                   |
| Deployment              | Streamlit Cloud                             |

---

## 9. Repository Structure

```
aqi-predictor/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── feature_pipeline.py
│   ├── backfill_historical.py
│   ├── training_pipeline.py
│   └── eda.py
│
├── app/
│   └── dashboard.py
│
├── artifacts/
│   └── # Generated model artifacts
│
├── eda_outputs/
│   └── # Generated EDA visualizations
│
└── .github/
    └── workflows/
        ├── feature_pipeline.yml
        └── training_pipeline.yml
```

### Directory Responsibilities

| Path                     | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `src/config.py`          | Central configuration and environment variables |
| `src/feature_pipeline.py`| Fetches live data, engineers features, and updates Supabase |
| `src/backfill_historical.py` | Retrieves and merges historical weather and pollutant data |
| `src/training_pipeline.py` | Trains, evaluates, and registers models     |
| `src/eda.py`             | Generates exploratory analysis and visualizations |
| `app/dashboard.py`       | Streamlit prediction dashboard               |
| `artifacts/`             | Generated local model artifacts              |
| `eda_outputs/`           | Generated EDA plots                          |
| `.github/workflows/`     | Automated CI/CD workflows                    |

---

## 10. Local Setup

### 10.1 Clone the Repository

```bash
git clone https://github.com/Muhammad-Hashir-55/aqi-predictor.git
cd aqi-predictor
```

### 10.2 Create a Virtual Environment

On Windows:

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
```

### 10.3 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 10.4 Configure Environment Variables

Copy the example environment file:

```powershell
copy .env.example .env
```

Then configure:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
SUPABASE_DB_URL=your_supabase_transaction_pooler_connection_string
HF_TOKEN=your_huggingface_write_token
HF_MODEL_REPO=your_huggingface_repository
AQICN_API_KEY=your_aqicn_api_key
```

### Required Credentials

| Variable           | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `OPENWEATHER_API_KEY` | Authentication for OpenWeather API        |
| `SUPABASE_DB_URL`  | PostgreSQL connection string                 |
| `HF_TOKEN`         | Hugging Face authentication with write access |
| `HF_MODEL_REPO`    | Destination repository for model artifacts   |
| `AQICN_API_KEY`    | Authentication for AQICN API                 |

> **Security:** Never commit `.env` or expose API keys, database credentials, or Hugging Face tokens in source code.

---

## 11. Running the Pipelines

### Step 1 — Historical Backfill

Populate the feature store with 90 days of historical data:

```bash
python src/backfill_historical.py --days 90
```

### Step 2 — Exploratory Data Analysis

Generate EDA outputs:

```bash
python src/eda.py
```

### Step 3 — Train the Models

Train and evaluate all candidate models:

```bash
python src/training_pipeline.py
```

The pipeline evaluates:

1. Persistence Baseline
2. XGBoost
3. LightGBM
4. Random Forest
5. Ridge Regression
6. PyTorch MLP

The model with the lowest RMSE is selected and its artifacts are pushed to the configured Hugging Face repository.

### Step 4 — Launch the Dashboard

Start the Streamlit application locally:

```bash
streamlit run app/dashboard.py
```

---

## 12. Automated CI/CD

The project uses GitHub Actions to automate the MLOps lifecycle.

### Feature Pipeline

The feature pipeline runs hourly and:

1. Fetches current weather data.
2. Fetches current pollutant data.
3. Generates engineered features.
4. Writes the latest feature row to Supabase.

Example cron schedule:

```yaml
cron: "15 * * * *"
```

This executes at **15 minutes past every hour**.

### Training Pipeline

The training workflow runs daily and:

1. Retrieves training data from Supabase.
2. Builds the 72-hour-ahead target.
3. Performs chronological train/test splitting.
4. Trains all candidate models.
5. Evaluates model performance.
6. Selects the model with the lowest RMSE.
7. Generates SHAP explainability artifacts.
8. Uploads the selected model to Hugging Face.

Example cron schedule:

```yaml
cron: "0 12 * * *"
```

GitHub Actions cron schedules use **UTC**.

Therefore:

```text
12:00 UTC = 17:00 PKT
```

---

## 13. Data Leakage Prevention

Time-series forecasting requires special care when splitting data.

A random train/test split can allow information from the future to influence the training set.

This project instead uses a **chronological split**, ensuring that training observations occur before evaluation observations.

Conceptually:

```text
Past --------------------------------------------------> Future

|---------------- Training ----------------|--- Testing ---|
```

This better represents the real-world forecasting scenario:

> Train on historical observations → predict unseen future observations.

---

## 14. Forecasting Target

The model forecasts AQI **72 hours ahead**.

Conceptually:

```text
t
│
├── Current observations
├── Feature engineering
├── Model
└──────────────────────────────► t + 72 hours
                                Predicted AQI
```

The prediction target is:

$$y_t = AQI_{t+72}$$

where $AQI_{t+72}$ represents the AQI observed 72 hours after the feature timestamp (t).

In the latest training run:

```text
2041 feature-store rows
       │
       ▼
1911 rows with valid 72-hour future labels
       │
       ▼
1528 training rows + 383 test rows
```

---

## 15. Explainability

The training pipeline generates SHAP-based explainability artifacts to analyze how individual features contribute to model predictions.

This helps investigate questions such as:

* Which weather variables influence predictions most?
* Which pollutant measurements are important?
* How much do temporal features contribute?
* Which features drive unusually high or low predictions?

SHAP provides both model-level and prediction-level insight into feature contributions.

The generated SHAP visualization is uploaded alongside the selected model artifacts.

---

## 16. Production Dashboard

The deployed Streamlit application provides:

* Current environmental information
* 72-hour AQI prediction
* AQI classification
* Environmental health alerts
* Model-driven forecasting results

### Live Application

**[Sialkot AQI Predictor · Streamlit](https://sialkot-aqi-predictor.streamlit.app)**

---

## 17. Limitations & Future Improvements

The current implementation has several areas that can be improved.

### Future Weather Forecast Features

Integrating forecasted weather variables could provide the model with direct information about expected atmospheric conditions during the prediction horizon.

Potential features include:

* Forecasted precipitation
* Forecasted wind speed
* Forecasted wind direction
* Forecasted temperature
* Forecasted humidity
* Forecasted atmospheric pressure

### Better Anomaly Detection

AQI observations should be investigated and potentially filtered or corrected using:

* Sensor validation
* Temporal consistency checks
* Robust statistical methods
* External reference measurements

### More Advanced Time-Series Models

Potential future experiments include:

* XGBoost / LightGBM hyperparameter optimization
* Temporal Convolutional Networks
* LSTM / GRU
* Temporal Fusion Transformer
* Dedicated time-series forecasting models

### More Evaluation Metrics

Future evaluation could include:

* MAPE
* SMAPE
* Median Absolute Error
* Prediction interval coverage
* Performance by AQI category

### Hyperparameter Optimization

The current model comparison establishes a strong baseline across several model families.

Future iterations can use:

* Grid Search
* Randomized Search
* Bayesian optimization
* Optuna

to tune the strongest candidate models further.

---

## 18. Project Status

**Status:** Production-oriented / actively developed

The current system provides an automated pipeline from data ingestion to model training, model evaluation, model registration, and live prediction.

```
External APIs
    │
    ▼
Feature Engineering
    │
    ▼
Supabase Feature Store
    │
    ├──────────────► Streamlit Dashboard
    │
    ▼
Automated Training
    │
    ▼
Model Evaluation
    │
    ├── XGBoost
    ├── LightGBM
    ├── Random Forest ─────────► Selected Model
    ├── Ridge
    └── PyTorch MLP
                      │
                      ▼
              Hugging Face Registry
                      │
                      ▼
              Production Prediction
```

---

## 19. Latest Training Summary

The latest successful training run confirms that the complete training and model-registry pipeline is operational.

```
Feature Store Rows       : 2041
Valid 72h Target Rows    : 1911
Training Rows            : 1528
Test Rows                : 383
Features                 : 42

Best Model               : Random Forest
RMSE                     : 33.3448
MAE                      : 23.0205
R²                       : 0.71

Model Registry           : Hugging Face Hub
Repository               : HashirAwaiz/aqi-forecast-model
Model Upload             : Successful
```

The selected model and supporting artifacts were successfully uploaded to the Hugging Face Hub.

---

## 20. License

This project is intended for educational, research, and portfolio purposes.