import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────
# Logging Setup
# ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────
# Streamlit Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="🌤️ Karachi AQI Forecast Dashboard",
    page_icon="💨",
    layout="wide"
)

# Apply Custom CSS Styling
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────
# Title and Subtitle
# ─────────────────────────────
st.title("🌤️ Karachi AQI Forecast Dashboard")
st.write("Real-time predicted AQI, interactive visualizations, and model insights powered by Hopsworks & Machine Learning.")

# ─────────────────────────────
# Environment & Hopsworks Connection
# ─────────────────────────────
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    st.error("❌ Missing Hopsworks API key!")
    st.stop()

try:
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    st.success("✅ Connected to Hopsworks successfully!")
except Exception as e:
    st.error(f"Connection failed: {e}")
    st.stop()

# ─────────────────────────────
# Load AQI Feature Data
# ─────────────────────────────
try:
    fg = fs.get_feature_group("karachi_aqi_us", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    st.info(f"📦 Loaded {len(df)} rows from feature group.")
except Exception as e:
    st.error(f"Error reading feature group: {e}")
    st.stop()

# ─────────────────────────────
# Load Latest Registered Model
# ─────────────────────────────
try:
    models = mr.get_models("rf_aqi_model")
    latest_model = max(models, key=lambda m: m.version)
    model_dir = latest_model.download()
    model_path = glob.glob(os.path.join(model_dir, "**/model.joblib"), recursive=True)[0]
    model = joblib.load(model_path)
    st.success(f"🧠 Loaded model version {latest_model.version}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ─────────────────────────────
# Prepare Input for Prediction
# ─────────────────────────────
drop_cols = ["time", "aqi_category", "us_aqi"]
feature_cols = [c for c in df.columns if c not in drop_cols]
latest_row = df.iloc[[-1]]
X_input = latest_row[feature_cols]

try:
    pred = float(model.predict(X_input)[0])
    latest_time = latest_row["time"].values[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ─────────────────────────────
# AQI Category Logic
# ─────────────────────────────
if pred <= 50:
    status, color = "Good", "#10b981"
elif pred <= 100:
    status, color = "Moderate", "#5b284a"
elif pred <= 150:
    status, color = "Unhealthy (Sensitive)", "#f97316"
elif pred <= 200:
    status, color = "Unhealthy", "#ef4444"
else:
    status, color = "Very Unhealthy", "#8b5cf6"

# ─────────────────────────────
# Metric Cards (Top Row)
# ─────────────────────────────
st.markdown("### 🌤️ Current AQI Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted AQI", f"{pred:.2f}")
col2.markdown(f"<div style='color:{color};font-weight:600;text-align:center'>{status}</div>", unsafe_allow_html=True)
col3.metric("Max AQI (24h)", f"{df['us_aqi'].tail(24).max():.2f}")
col4.metric("Min AQI (24h)", f"{df['us_aqi'].tail(24).min():.2f}")

st.caption(f"📅 Latest Data Timestamp: {latest_time}")

# ─────────────────────────────
# AQI Trend Over Time
# ─────────────────────────────
st.markdown("---")
st.subheader("📆 AQI Trend (Last 150 Hours)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["time"].tail(150), df["us_aqi"].tail(150), marker="o", color="dodgerblue")
ax.set_title("AQI Trend Over Time")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("AQI")
plt.xticks(rotation=45)
st.pyplot(fig)

# ─────────────────────────────
# AQI Distribution
# ─────────────────────────────
st.subheader("📊 AQI Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["us_aqi"], bins=20, color="skyblue", kde=True, ax=ax)
ax.set_xlabel("AQI")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ─────────────────────────────
# Correlation Heatmap
# ─────────────────────────────
st.subheader("🔥 Feature Correlation Heatmap")
corr = df[feature_cols + ["us_aqi"]].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# ─────────────────────────────
# Feature Importance
# ─────────────────────────────
st.subheader("🌲 Feature Importance (Model Insights)")
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Top Feature Importance in AQI Prediction")
    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")

# ─────────────────────────────
# Actual vs Predicted AQI
# ─────────────────────────────
st.subheader("📈 Actual vs Predicted AQI (Last 60 Records)")
compare_df = df.tail(60).copy()
compare_df["Predicted_AQI"] = model.predict(compare_df[feature_cols])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(compare_df["time"], compare_df["us_aqi"], label="Actual AQI", color="skyblue", marker="o")
ax.plot(compare_df["time"], compare_df["Predicted_AQI"], label="Predicted AQI", color="orange", marker="x")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("AQI")
ax.set_title("Actual vs Predicted AQI")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# ─────────────────────────────
# Expandable Section - Recent Raw Data
# ─────────────────────────────
with st.expander("📋 View Latest Data Sample"):
    st.dataframe(df.tail(10))

