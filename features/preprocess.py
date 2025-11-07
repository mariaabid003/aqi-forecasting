import hopsworks
import hsfs
import pandas as pd
import os
from dotenv import load_dotenv

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
    raise ValueError("Missing HOPSWORKS_API_KEY or HOPSWORKS_PROJECT in .env")

# -------------------------------------------------------
# Connect to Hopsworks
# -------------------------------------------------------
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

# -------------------------------------------------------
# Load Feature Group
# -------------------------------------------------------
print("📥 Loading feature group 'karachi_aqi_us'...")
fg = fs.get_feature_group("karachi_aqi_us", version=1)
df = fg.read()
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from karachi_aqi_us.")

# -------------------------------------------------------
# Basic Preprocessing Checks (Before)
# -------------------------------------------------------
print("\n📊 Data Overview (Before Cleaning):")
print(df.info())

print("\n🔍 Missing values before cleaning:")
print(df.isna().sum())

# -------------------------------------------------------
# Handle Missing Values (Option 2: Forward Fill)
# -------------------------------------------------------
cols_to_fill = ["aqi_pm10", "aqi_o3"]

for col in cols_to_fill:
    if col in df.columns:
        before = df[col].isna().sum()
        df[col].fillna(method="ffill", inplace=True)
        after = df[col].isna().sum()
        print(f"🧹 Filled NaNs in '{col}' — before: {before}, after: {after}")

# -------------------------------------------------------
# Duplicate Check
# -------------------------------------------------------
duplicate_count = df.duplicated().sum()
print(f"\n🧾 Total duplicate rows: {duplicate_count}")

# -------------------------------------------------------
# Validation Summary
# -------------------------------------------------------
print("\n✅ Missing values after cleaning:")
print(df.isna().sum())

print("\n🎉 Preprocessing complete — dataset cleaned successfully!")