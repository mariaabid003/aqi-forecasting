import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
from datetime import timedelta
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_rf_model(mr):
    models = mr.get_models(name="rf_aqi_model")
    if not models:
        log.error("No RF AQI model found in Hopsworks.")
        exit(1)
    latest_model = max(models, key=lambda m: m.version)
    model_dir = latest_model.download()
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    log.info(f"Loaded RandomForest model version: {latest_model.version}")
    return model

def forecast_with_trend(model, df, days=3, trend_days=20):
    last_row = df.iloc[[-1]].copy()
    drop_cols = ["time", "aqi_category", "us_aqi"]
    X_input = last_row.drop(columns=drop_cols, errors="ignore")
    X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)
    last_n = df.tail(trend_days)
    slope = (last_n["us_aqi"].iloc[-1] - last_n["us_aqi"].iloc[0]) / (trend_days - 1)
    forecasts = []
    for i in range(1, days + 1):
        pred = float(model.predict(X_input)[0])
        pred_trend = pred + slope * i
        forecast_date = last_row["time"].values[0] + np.timedelta64(i, "D")
        forecasts.append({
            "date": pd.Timestamp(forecast_date),
            "predicted_aqi": round(pred_trend, 2)
        })
    return forecasts

def upload_forecast_to_hopsworks(forecast):
    log.info("Connecting to Hopsworks...")
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project_name or not api_key:
        raise EnvironmentError("Missing HOPSWORKS_PROJECT or HOPSWORKS_API_KEY in environment.")
    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()
    df_pred = pd.DataFrame(forecast)
    df_pred["event_time"] = df_pred["date"]
    feature_group = fs.get_or_create_feature_group(
        name="karachi_aqi_predictions",
        version=1,
        description="3-day AQI predictions generated from RF model",
        primary_key=["date"],
        event_time="event_time"
    )
    feature_group.insert(df_pred, write_options={"wait_for_job": False})
    log.info("Successfully uploaded predictions to Hopsworks.")

def get_forecast_data():
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    fg = fs.get_feature_group("karachi_aqi_us", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    log.info(f"Loaded {len(df)} rows from Feature Group: karachi_aqi_us")
    model = load_rf_model(mr)
    forecast = forecast_with_trend(model, df, days=3, trend_days=20)
    return df, forecast

def main():
    df, forecast = get_forecast_data()
    log.info("3-Day AQI Forecast with 20-day Trend:")
    for f in forecast:
        log.info(f"Date {f['date']} → AQI {f['predicted_aqi']}")
    upload_forecast_to_hopsworks(forecast)
    log.info("Predictions uploaded successfully.")

if __name__ == "__main__":
    main()
