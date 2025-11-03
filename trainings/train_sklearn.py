import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
from dotenv import load_dotenv
import logging

# ------------------------------------------------
# ✅ Setup logging
# ------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# ✅ Load environment variables
# ------------------------------------------------
load_dotenv()

# ------------------------------------------------
# ✅ Connect to Hopsworks
# ------------------------------------------------
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)
fs = project.get_feature_store()
mr = project.get_model_registry()

logger.info(f"✅ Logged in to project: {project.get_url()}")

# ------------------------------------------------
# ✅ Load feature group
# ------------------------------------------------
fg = fs.get_feature_group("karachi_aqi_us", version=1)
df = fg.read()
logger.info(f"✅ Loaded {len(df)} rows from Feature Group: karachi_aqi_us")

# ------------------------------------------------
# ✅ Prepare data
# ------------------------------------------------
df = df.dropna(subset=["us_aqi"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

target_col = "us_aqi"
drop_cols = ["time", "aqi_category"]

X = df.drop(columns=drop_cols + [target_col], errors="ignore")
y = df[target_col]

# ------------------------------------------------
# ✅ Time-based split (last 10 days = test)
# ------------------------------------------------
split_time = df["time"].max() - pd.Timedelta(days=10)
X_train = df[df["time"] <= split_time].drop(columns=drop_cols + [target_col])
y_train = df[df["time"] <= split_time][target_col]
X_test = df[df["time"] > split_time].drop(columns=drop_cols + [target_col])
y_test = df[df["time"] > split_time][target_col]

logger.info(f"🧠 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ------------------------------------------------
# ✅ Train model (tuned for ~80% R²)
# ------------------------------------------------
model = RandomForestRegressor(
    n_estimators=120,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)



logger.info("🤖 Training RandomForestRegressor...")
model.fit(X_train, y_train)

# After model.fit(X_train, y_train)
model.feature_names_in_ = X_train.columns.tolist()

# ------------------------------------------------
# ✅ Evaluate
# ------------------------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logger.info("✅ Model performance on unseen (future) data:")
logger.info(f"   RMSE: {rmse:.2f}")
logger.info(f"   MAE : {mae:.2f}")
logger.info(f"   R²  : {r2:.2f}")

# ------------------------------------------------
# ✅ Save model (same pattern as your friend)
# ------------------------------------------------
MODEL_DIR = "models/rf_aqi_model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(model, MODEL_PATH)
logger.info(f"✅ Model saved to {MODEL_PATH}")

# ------------------------------------------------
# ✅ Upload to Hopsworks Model Registry
# ------------------------------------------------
logger.info("🚀 Uploading model to Hopsworks Model Registry...")

model_meta = mr.python.create_model(
    name="rf_aqi_model",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="Random Forest model for Karachi AQI forecasting (target ~80% accuracy)"
)

# ✅ Option 1: Upload the entire folder
model_meta.save(MODEL_DIR)

logger.info("🎉 Model successfully uploaded to Hopsworks Model Registry!")
logger.info(f"🌐 Explore it here: {project.get_url()}/p/{project.id}/models")
logger.info("🏁 Training pipeline completed successfully.") 