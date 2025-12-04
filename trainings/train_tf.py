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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)

fs = project.get_feature_store()
mr = project.get_model_registry()
log.info(f"Logged in to project: {project.get_url()}")

fg = fs.get_feature_group("karachi_aqi_us", version=1)

# 🔥 FORCE HIVE MODE – BYPASS HUDI + ARROW COMPLETELY
df = fg.read(
    read_options={
        "use_hive": True,
        "use_arrow_flight": False
    }
)

log.info(f"Loaded {len(df)} rows from karachi_aqi_us")
