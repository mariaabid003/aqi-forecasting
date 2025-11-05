#  **Pearls AQI Forecasting Project**

This project predicts the Air Quality Index (AQI) in Karachi for the next 3 days using a 100% serverless machine learning stack. It features a complete end-to-end pipeline
using historical air pollution and meteorological data, providing actionable insights for environmental awareness and health safety.

---

##  **Table of Contents**

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#️-setup--installation)
- [Environment Variables](#-environment-variables)
- [How it Works](#-how-it-works)
- [Model Training](#-model-training)
- [Real-time Predictions](#️-real-time-predictions)
- [Streamlit Dashboard](#-dashboard)
- [CI/CD](#-cicd)


---

## **Project Overview**

This project predicts the Air Quality Index (AQI) in Karachi for the next 3 days using historical air pollution and weather data.  
It leverages data pipelines, feature engineering, and machine learning models to deliver real-time, explainable predictions.

###  **Key Objectives**

- Automated data collection from external APIs (Open-Meteo, AQICN).  
- Feature engineering including time-based features and pollutant trend analysis.  
- ML pipelines for model training and evaluation (Random Forest & Dense Neural Networks).  
- Real-time AQI predictions displayed via an interactive Streamlit dashboard.  
- Deployment on a 100% serverless architecture using Hopsworks Feature Store and Model Registry.  

---

##  **Features**

###  **Feature Pipeline**
- Fetches raw air quality and weather data.  
- Computes derived features (e.g., AQI change rate, temperature-humidity index).  
- Stores processed features in Hopsworks Feature Store for training and inference.

###  **Historical Data Backfill**
- Generates datasets for training using minimum 300 days of data.  
- Ensures temporal consistency and feature completeness.

###  **Training Pipeline
- Trains ML models: **Random Forest**, **Ridge Regression**, and **TensorFlow Dense Neural Network**.  
- Evaluates models with **RMSE**, **MAE**, and **R²** metrics.  
- Uploads trained models to **Hopsworks Model Registry** for versioning.

###  **Prediction Pipeline**
- Fetches the latest features and computes the AQI forecast for the **next 3 days**.  
- Uploads predictions to a new Hopsworks Feature Group (`karachi_aqi_predictions`).

###  **Dashboard**
An **interactive Streamlit dashboard** visualizes:
- 🟢 Current AQI with pollutant details  
- 📅 3-day forecast cards  
- 📊 Actual vs Predicted AQI chart  
- 📈 Trend chart & correlation heatmap  

###  **Explainability**
- Sends alerts for hazardous AQI levels (e.g., via notifications or color-coded UI).

---

## 🧠 **Technology Stack**

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Programming & ML** | Python, pandas, numpy, scikit-learn, TensorFlow |
| **Feature Store & Registry** | Hopsworks |
| **Automation / MLOps** | Apache Airflow / GitHub Actions |
| **Web Dashboard** | Streamlit, Flask |
| **APIs** | AQICN, OpenWeather, Open-Meteo |
| **Explainability & Visualization** | SHAP, Plotly, Altair, Seaborn, Matplotlib |
| **Utilities** | python-dotenv, joblib, tqdm, rich |

---

## 🗂️ **Project Structure**

```bash
aqi_forecast/
├── .github/
│   └── workflows/
│       └── pipeline.yaml          # CI/CD workflow for GitHub Actions
│
├── app/
│   ├── dashboard.py               # Streamlit dashboard for visualization
│   └── style.css                  # Custom CSS for dashboard styling
│
├── data/
│   ├── raw_openmeteo/             # Raw weather data fetched from Open-Meteo API
│   ├── features/                  # Engineered features ready for training
│   └── predictions/               # Stored forecast results
│
├── features/
│   ├── backfill.py                # Historical data backfill and feature creation
│   ├── compute_aqi.py             # AQI computation logic (US AQI scale)
│   └── live_aqi.py                # Fetches and updates live AQI readings
│
├── trainings/
│   ├── train_sklearn.py           # Random Forest and Ridge Regression training
│   ├── train_tf.py                # TensorFlow Dense Neural Network training
│   └── predict.py                 # Forecasting next 3-day AQI using trained models
│
├── eda.ipynb                      # Exploratory Data Analysis notebook
├── requirements.txt               # Python dependencies
└── .env                           # Environment variables (API keys, configs)
```

---
## ⚙️ **Setup Instructions**

###  **1. Clone Repository***
```bash
git clone https://github.com/<your-username>/AQI_Forecast.git
cd AQI_Forecast
```

###  **2. Create Virtual Environment**

```bash
conda create aqi-py310
source venv/bin/activate       # Mac/Linux
conda activate aqi-py310         # Windows

```
### 📦 **3. Install Dependencies**
``bash
pip install --upgrade pip
pip install -r requirements.txt


### 📦 **4. Configure .env File**
```bash
HOPSWORKS_HOST=your host
AQICN_TOKEN=your_aqicn_token
HOPSWORKS_API_KEY=hopsworks_api_key
HOPSWORKS_PROJECT=name_of_the_project
HOPSWORKS_PROJECT_ID=project_id
```

### 📦 **5. Authenticate Hopsworks**
```bash
python -m hopsworks.login
```

---

##  **How it Works**

### 🧾 **Data Ingestion**
- `backfill_data.py` fetches historical AQI and weather data for Karachi  
- Data is uploaded to the Hopsworks Feature Store.

---

### 🌫️ **AQI Computation**
- `compute_aqi.py` calculates **PM2.5**, **PM10**, **O₃ AQI**, and overall **US AQI**  
- Categories include: *Good*, *Moderate*, *Unhealthy (SG)*, *Unhealthy*, *Very Unhealthy*, *Hazardous*

---

###  **Model Training**

####  **Random Forest**
- Tuned for 80% R² on unseen data  
- Features: pollutant concentrations, temperature, humidity, wind speed  

####  **Dense Neural Network**
- 2 hidden layers (64, 32 units) with Dropout
- Uses StandardScaler for normalization  
- Implements Early Stopping and Learning Rate Scheduler
- Metrics: RMSE, MAE, R²

---

###  **Real-time Predictions**
- Uses latest observed AQI and 20-day trend to forecast 3 days ahead
- Predictions automatically uploaded to Hopsworks Feature Store

---

###  **Dashboard**
- Built with **Streamlit**
- Displays:
  - 🌫️ Current AQI and pollutants  
  - 📅 3-day forecast  
  - 📈 Actual vs Predicted AQI chart  
  - 📉 Trend and correlation heatmap  
- Styled using **style.css**

---

###  **CI/CD**
- GitHub Actions automate:
  -  Hourly feature pipeline runs  
  -  Daily model training pipeline  
  -  Auto deployment to Streamlit Cloud

---
##  **Key Outputs**

- `trainings/predict.py` → Today's forecast + 3-day AQI forecast with alerts  
- `eda.ipynb` → Generated EDA visuals  
- `models/` → Saved model artifacts  
- `aqi_features` → Feature Store on Hopsworks

---

##  **Dashboard Preview**

- **Today’s AQI Summary** — color-coded & mood-based  
- **Next 3-Day Forecast** — with interactive charts  
- **EDA Visuals** — trends, correlations, and feature importance (complete `eda_outputs`)  

**Live Dashboard:**  
 *Pearls AQI Predictor — Streamlit App*

**Run locally:**
``bash
streamlit run dashboard/dashboard.py

---
##  **Future Enhancements**

- Add SHAP/LIME explainability  
- Integrate data validation using Great Expectations
- Extend to multi-city forecasting

---
## 🙌 **Acknowledgments**

- **AQICN** — Air Quality API  
- **OpenWeather** — Weather API  
- **Open-Meteo** — Historical Weather Data  
- **Hopsworks** — Feature Store & Model Registry  
- **Streamlit** — Dashboard Framework

---

<div align="center">

###  **Maria Abid**  
**Data Engineer**  
*mariaabid003@gmail.com*  

 **Pearls AQI Predictor (2025)**  

</div>











  
