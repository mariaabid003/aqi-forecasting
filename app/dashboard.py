import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pathlib

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="🌆 Karachi AQI Forecast Dashboard",
    page_icon="💨",
    layout="wide"
)

# -------------------------------
# Apply External CSS
# -------------------------------
def local_css(file_name):
    css_path = pathlib.Path(__file__).parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Could not find {file_name}. Make sure it’s in the same folder as dashboard.py")

local_css("style.css")

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>🌃 Karachi AQI Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-driven air quality forecasting with RandomForest & real-time Hopsworks data</p>", unsafe_allow_html=True)

# -------------------------------
# Hopsworks Connection
# -------------------------------
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not API_KEY or not PROJECT:
    st.error("❌ Missing Hopsworks credentials.")
    st.stop()

try:
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    st.success("✅ Connected to Hopsworks successfully.")
except Exception as e:
    st.error(f"Connection failed: {e}")
    st.stop()

# -------------------------------
# Load AQI Data
# -------------------------------
try:
    fg = fs.get_feature_group("karachi_aqi_us", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    st.info(f"📦 Loaded {len(df)} records from Feature Group")
except Exception as e:
    st.error(f"Failed to read feature group: {e}")
    st.stop()

# -------------------------------
# AQI Category Helper
# -------------------------------
def aqi_category(aqi):
    if aqi <= 50: return "Good", "#22c55e"
    elif aqi <= 100: return "Moderate", "#facc15"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "#fb923c"
    elif aqi <= 200: return "Unhealthy", "#ef4444"
    else: return "Very Unhealthy", "#a855f7"

# -------------------------------
# Small helper: find pollutant columns if present
# -------------------------------
def find_pollutants(row):
    # common pollutant name variations -> label to column mapping
    candidates = {
        "PM2.5": ["pm2_5", "pm25", "pm_2_5", "pm2.5", "pm_25"],
        "PM10": ["pm10", "pm_10"],
        "SO2": ["so2", "s02"],
        "O3": ["o3"],
        "CO": ["co"],
        "NO2": ["no2", "n02"]
    }
    found = {}
    for label, names in candidates.items():
        for n in names:
            if n in row.index:
                try:
                    val = row[n]
                    if pd.notnull(val):
                        found[label] = float(val)
                        break
                except Exception:
                    pass
    return found

# -------------------------------
# Show Today's AQI (Mint-style card)
# -------------------------------
latest = df.tail(1).iloc[0]
latest_time_for_date = latest["time"].strftime("%A, %b %d")  # date only, no "updated"
latest_aqi = latest["us_aqi"]
cat, color = aqi_category(latest_aqi)

# Pollutant values (optional display)
pollutants = find_pollutants(latest)

st.markdown("<h2 class='section-title'>📍 Live AQI - Karachi</h2>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='today-aqi-card'>
      <div class='today-left'>
        <div class='aqi-circle'>{int(round(latest_aqi))}</div>
      </div>
      <div class='today-right'>
        <div class='city-name'>Karachi</div>
        <div class='today-date'>{latest_time_for_date}</div>
        <div class='today-cat' style='color:{color};'>{cat}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# If pollutant info available, show small tiles (keeps visual similar to reference)
if len(pollutants) > 0:
    pollutant_html = "<div class='pollutant-row'>"
    for k, v in pollutants.items():
        pollutant_html += f"""
        <div class='poll-tile'>
            <div class='pt-label'>{k}</div>
            <div class='pt-value'>{int(round(v))}</div>
        </div>
        """
    pollutant_html += "</div>"
    st.markdown(pollutant_html, unsafe_allow_html=True)

# -------------------------------
# Load 3-Day Forecast
# -------------------------------
try:
    pred_fg = fs.get_feature_group("karachi_aqi_predictions", version=1)
    pred_df = pred_fg.read()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date").reset_index(drop=True)
    forecast = pred_df.tail(3).to_dict(orient="records")
    st.success("✅ Loaded 3-day forecast from Hopsworks")
except Exception as e:
    st.error(f"Failed to load predictions from Hopsworks: {e}")
    st.stop()

# -------------------------------
# Display Forecast (unchanged logic; styled)
# -------------------------------
st.markdown("<h2 class='section-title'>📊 AQI Forecast (Next 3 Days)</h2>", unsafe_allow_html=True)
cols = st.columns(3)

for i, row in enumerate(forecast):
    aqi = row["predicted_aqi"]
    date_label = row["date"].strftime("%a, %b %d")
    cat, color = aqi_category(aqi)

    cols[i].markdown(
        f"""
        <div class='aqi-card' style='border-top: 6px solid {color};'>
            <h3>{date_label}</h3>
            <p class='aqi-value'>{aqi:.2f} AQI</p>
            <p class='aqi-cat' style='color:{color};'>{cat}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Load Model (unchanged)
# -------------------------------
try:
    models = mr.get_models(name="rf_aqi_model")
    latest_model = max(models, key=lambda m: m.version)
    model_dir = latest_model.download()
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    st.info(f"🧠 Using RandomForest model version {latest_model.version}")
except Exception as e:
    st.warning(f"⚠️ Model not loaded, skipping prediction-based charts: {e}")
    model = None

# -------------------------------
# Actual vs Predicted Chart
# -------------------------------
if model is not None:
    st.markdown("<h2 class='section-title'>🎯 Actual vs Predicted AQI</h2>", unsafe_allow_html=True)
    try:
        recent = df.tail(50).copy()
        X_recent = recent.drop(columns=["time", "aqi_category", "us_aqi"], errors="ignore")
        X_recent = X_recent.reindex(columns=model.feature_names_in_, fill_value=0)
        recent["Predicted_AQI"] = model.predict(X_recent)

        fig = px.line(
            recent,
            x="time",
            y=["us_aqi", "Predicted_AQI"],
            labels={"value": "AQI", "time": "Date"},
            title="Actual vs Predicted AQI (Last 50 Records)",
            color_discrete_sequence=["#6e7c7c", "#8ca6a3"]
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate Actual vs Predicted chart: {e}")

# -------------------------------
# AQI Trend
# -------------------------------
st.markdown("<h2 class='section-title'>📈 AQI Trend (Last 100 Hours)</h2>", unsafe_allow_html=True)
fig = px.line(df.tail(100), x="time", y="us_aqi", markers=True,
              color_discrete_sequence=["#6e7c7c"],
              title="Recent AQI Trend")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Correlation Heatmap
# -------------------------------
st.markdown("<h2 class='section-title'>🔥 Correlation Heatmap (Last 30 Days)</h2>", unsafe_allow_html=True)

try:
    last_30_df = df[df["time"] >= (df["time"].max() - pd.Timedelta(days=30))]
    numeric_df = last_30_df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="Purples", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap (Last 30 Days)", fontsize=14)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric data to compute correlation heatmap.")
except Exception as e:
    st.error(f"⚠️ Could not generate heatmap: {e}")

# -------------------------------
# Data Viewer
# -------------------------------
with st.expander("📋 View Latest Data Sample"):
    st.dataframe(df.tail(10))
