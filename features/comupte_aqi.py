import hopsworks
import hsfs
import pandas as pd
import os

# -------------------------------------------------------
# ✅ Load environment variables
# -------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY or HOPSWORKS_PROJECT in .env")

# -------------------------------------------------------
# ✅ Connect to Hopsworks
# -------------------------------------------------------
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

# -------------------------------------------------------
# ✅ Load the backfill feature group
# -------------------------------------------------------
print("📥 Loading feature group 'karachi_aqi_backfill'...")
fg = fs.get_feature_group("karachi_aqi_backfill", version=1)
df = fg.read()
print(f"✅ Loaded {len(df)} rows from karachi_aqi_backfill.")

# -------------------------------------------------------
# ✅ Define AQI calculation functions
# -------------------------------------------------------
def calculate_aqi(concentration, breakpoints):
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            return ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
    return None

def aqi_pm25(c):
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    return calculate_aqi(c, bps)

def aqi_pm10(c):
    bps = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]
    return calculate_aqi(c, bps)

def aqi_o3(c):
    ppm = c / 2000  # approximate conversion µg/m³ → ppm
    bps = [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ]
    return calculate_aqi(ppm, bps)

def aqi_category(aqi):
    if pd.isna(aqi): return None
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy (SG)"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

# -------------------------------------------------------
# ✅ Compute AQI values
# -------------------------------------------------------
print("📊 Computing US AQI values...")

df["aqi_pm25"] = df["pm2_5"].apply(aqi_pm25)
df["aqi_pm10"] = df["pm10"].apply(aqi_pm10)
df["aqi_o3"] = df["ozone"].apply(aqi_o3)
df["us_aqi"] = df[["aqi_pm25", "aqi_pm10", "aqi_o3"]].max(axis=1)
df["aqi_category"] = df["us_aqi"].apply(aqi_category)

print("✅ Sample with computed AQI:")
print(df[["time", "pm2_5", "pm10", "ozone", "us_aqi", "aqi_category"]].head())

# -------------------------------------------------------
# ✅ Upload to new feature group
# -------------------------------------------------------
print("🚀 Creating new feature group: 'karachi_aqi_us'...")

fg_us = fs.get_or_create_feature_group(
    name="karachi_aqi_us",
    version=1,
    description="Karachi AQI backfill data with computed US AQI and categories",
    primary_key=["time"],
    event_time="time"
)

fg_us.insert(df, write_options={"wait_for_job": True})

print("✅ Successfully uploaded AQI-enhanced dataset to Hopsworks.")
print("🌐 View it at:")
print(f"{project.get_url()}/p/{project.project_id}/fs/{fs.id}/fg/{fg_us.id}")

print("Connection closed.")
