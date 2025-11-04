import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_ID = int(os.getenv("HOPSWORKS_PROJECT_ID"))  # Use ID instead of name

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_ID)
fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_us", version=1)

df = fg.read()
print("✅ Rows in FG:", len(df))
print(df.tail())
