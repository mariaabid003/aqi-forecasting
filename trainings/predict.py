import os
import pandas as pd
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import logging
from dotenv import load_dotenv
import joblib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_COLS = []  # Will be set dynamically from the model's features

def load_model(mr):
    """Load the latest RandomForest model from Hopsworks"""
    models = mr.get_models(name="rf_aqi_model")
    if not models:
        log.error("No Random Forest AQI model found in Hopsworks.")
        exit(1)

    latest_model = max(models, key=lambda m: m.version)
    model_dir = latest_model.download()
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model, latest_model.version

def main():
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # -----------------------------
    # Load data
    # -----------------------------
    fg = fs.get_feature_group("karachi_aqi_us", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    log.info(f"✅ Loaded {len(df)} rows from Feature Group: karachi_aqi_us")

    # -----------------------------
    # Take the last row (latest)
    # -----------------------------
    last_row = df.iloc[[-1]].copy()
    drop_cols = ["time", "aqi_category", "us_aqi"]
    X_input = last_row.drop(columns=drop_cols, errors="ignore")

    global FEATURE_COLS
    FEATURE_COLS = X_input.columns.tolist()

    # Ensure all required columns exist
    for col in FEATURE_COLS:
        if col not in X_input.columns:
            X_input[col] = 0

    # -----------------------------
    # Load model
    # -----------------------------
    model, version = load_model(mr)

    # -----------------------------
    # Predict today's AQI
    # -----------------------------
    predicted_aqi = float(model.predict(X_input)[0])
    predicted_time = last_row["time"].values[0]

    result = {
        "predicted_aqi": predicted_aqi,
        "predicted_at": pd.Timestamp(predicted_time).isoformat(),
        "model_version": int(version)
    }

    log.info(f"Predicted AQI: {predicted_aqi:.2f} (model v{version})")
    log.info("Prediction completed successfully.")
    print(result)

if __name__ == "__main__":
    main()
