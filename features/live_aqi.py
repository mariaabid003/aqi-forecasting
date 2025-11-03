import os
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY. Set it in .env or GitHub Secrets!")

# -----------------------------
# Config
# -----------------------------
LAT, LON = 24.8607, 67.0011  # Karachi
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# -----------------------------
# AQI calculation functions
# -----------------------------
def calculate_aqi(concentration, breakpoints):
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            return ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
    return None

def aqi_pm25(c):
    bps = [
        (0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,150.4,151,200),(150.5,250.4,201,300),
        (250.5,350.4,301,400),(350.5,500.4,401,500)
    ]
    return calculate_aqi(c, bps)

def aqi_pm10(c):
    bps = [
        (0,54,0,50),(55,154,51,100),(155,254,101,150),
        (255,354,151,200),(355,424,201,300),
        (425,504,301,400),(505,604,401,500)
    ]
    return calculate_aqi(c, bps)

def aqi_o3(c):
    ppm = c / 2000  # µg/m³ → ppm
    bps = [
        (0.000,0.054,0,50),(0.055,0.070,51,100),
        (0.071,0.085,101,150),(0.086,0.105,151,200),
        (0.106,0.200,201,300)
    ]
    return calculate_aqi(ppm, bps)

def aqi_category(aqi):
    if pd.isna(aqi):
        return None
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy (SG)"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

# -----------------------------
# Fetch latest air quality
# -----------------------------
def fetch_latest_air_quality():
    now = datetime.utcnow()
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": now.strftime("%Y-%m-%d"),
        "end_date": now.strftime("%Y-%m-%d"),
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    }
    r = requests.get(AIR_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").tail(1)

# -----------------------------
# Fetch latest weather
# -----------------------------
def fetch_latest_weather():
    now = datetime.utcnow()
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": now.strftime("%Y-%m-%d"),
        "end_date": now.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }
    r = requests.get(WEATHER_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").tail(1)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_aqi_us", version=1)

    print("🌍 Fetching latest air quality and weather data...")
    df_air = fetch_latest_air_quality()
    df_weather = fetch_latest_weather()
    df = pd.merge(df_air, df_weather, on="time", how="inner")

    # Compute AQI
    df["aqi_pm25"] = df["pm2_5"].apply(aqi_pm25)
    df["aqi_pm10"] = df["pm10"].apply(aqi_pm10)
    df["aqi_o3"] = df["ozone"].apply(aqi_o3)
    df["us_aqi"] = df[["aqi_pm25","aqi_pm10","aqi_o3"]].max(axis=1)
    df["aqi_category"] = df["us_aqi"].apply(aqi_category)

    # 👇 Force a unique timestamp by adding microseconds (prevents overwrite)
    df["time"] = df["time"] + pd.to_timedelta(np.random.randint(1, 999999), unit="us")

    # Append to feature group
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Appended latest AQI data for {df['time'].iloc[0]} to Hopsworks.")
