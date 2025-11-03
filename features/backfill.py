import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# ✅ Config
# -----------------------------
LAT, LON = 24.8607, 67.0011  # Karachi
DAYS = 30


AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# -----------------------------
# ✅ Fetch Air Quality Data
# -----------------------------
def fetch_air_quality():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=DAYS)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join([
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone"
        ])
    }

    print("🌤 Fetching air quality data from Open-Meteo...")
    r = requests.get(AIR_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df_air = pd.DataFrame(data["hourly"])
    df_air["time"] = pd.to_datetime(df_air["time"])
    print(f"✅ Air quality rows: {len(df_air)}")
    return df_air


# -----------------------------
# ✅ Fetch Weather Data
# -----------------------------
def fetch_weather():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=DAYS)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m"
        ])
    }

    print("🌦 Fetching weather data from Open-Meteo...")
    r = requests.get(WEATHER_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df_weather = pd.DataFrame(data["hourly"])
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    print(f"✅ Weather rows: {len(df_weather)}")
    return df_weather


# -----------------------------
# ✅ Merge & Upload
# -----------------------------
def upload_to_hopsworks(df):
    print("🔗 Connecting to Hopsworks...")

    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        raise EnvironmentError("❌ Missing HOPSWORKS_PROJECT or HOPSWORKS_API_KEY in environment.")

    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name="karachi_aqi_backfill",
        version=1,
        description="Backfilled AQI + weather data for Karachi",
        primary_key=["time"],
        event_time="time"
    )

    feature_group.insert(df, write_options={"wait_for_job": False})
    print("✅ Successfully uploaded backfilled data to Hopsworks.")


# -----------------------------
# ✅ Main
# -----------------------------
if __name__ == "__main__":
    try:
        df_air = fetch_air_quality()
        df_weather = fetch_weather()

        df = pd.merge(df_air, df_weather, on="time", how="inner")
        print(f"✅ Combined total of {len(df)} rows.")

        if not df.empty:
            upload_to_hopsworks(df)
        else:
            print("⚠️ No data merged. Exiting.")
    except Exception as e:
        print(f"❌ {e}")
    finally:
        print("Connection closed.")
