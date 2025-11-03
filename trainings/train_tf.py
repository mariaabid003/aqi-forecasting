# trainings/train_tf.py
import os
import numpy as np
import pandas as pd
import hopsworks
import joblib
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import tempfile

# ------------------------------------------------
# ✅ Setup logging
# ------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ------------------------------------------------
# ✅ Load environment & connect to Hopsworks
# ------------------------------------------------
load_dotenv()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)
fs = project.get_feature_store()
mr = project.get_model_registry()
log.info(f"✅ Logged in to project: {project.get_url()}")

# ------------------------------------------------
# ✅ Load Feature Group
# ------------------------------------------------
fg = fs.get_feature_group("karachi_aqi_us", version=1)
df = fg.read()
log.info(f"✅ Loaded {len(df)} rows from karachi_aqi_us")

# ------------------------------------------------
# ✅ Clean & select features
# ------------------------------------------------
df = df.dropna(subset=["us_aqi"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

FEATURES = [
    "pm2_5",
    "pm10",
    "ozone",
    "sulphur_dioxide",
    "nitrogen_dioxide",
    "carbon_monoxide",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]
TARGET = "us_aqi"

df = df[["time"] + FEATURES + [TARGET]]

# ------------------------------------------------
# ✅ Time-based train-test split (last 10 days = test)
# ------------------------------------------------
split_time = df["time"].max() - pd.Timedelta(days=10)
train_df = df[df["time"] <= split_time]
test_df = df[df["time"] > split_time]

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values.reshape(-1, 1)
X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values.reshape(-1, 1)

log.info(f"🧠 Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ------------------------------------------------
# ✅ Scale data
# ------------------------------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# ------------------------------------------------
# ✅ Build simple Dense NN model
# ------------------------------------------------
model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
log.info("🤖 Model compiled successfully.")

# ------------------------------------------------
# ✅ Train model
# ------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.1,
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------
# ✅ Evaluate model
# ------------------------------------------------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_scaled)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

log.info("📈 Model Performance on Test Data:")
log.info(f"RMSE: {rmse:.2f}")
log.info(f"MAE : {mae:.2f}")
log.info(f"R²  : {r2:.4f}")

# ------------------------------------------------
# ✅ Upload model to Hopsworks
# ------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "tf_dense_model.keras")
    model.save(model_path)
    joblib.dump(x_scaler, os.path.join(tmpdir, "x_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(tmpdir, "y_scaler.joblib"))

    model_meta = mr.python.create_model(
        name="tf_dense_aqi_model",
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        description=f"Dense Neural Network for Karachi AQI (R²={r2:.2f})"
    )
    model_meta.save(tmpdir)

log.info("🎉 Model uploaded to Hopsworks successfully!")
log.info(f"🌐 Explore at: {project.get_url()}/p/{project.id}/models")
