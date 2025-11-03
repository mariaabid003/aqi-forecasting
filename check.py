import requests
import json

latitude = 24.8607
longitude = 67.0011

url = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={latitude}&longitude={longitude}"
    "&hourly=pm10,pm2_5,carbon_monoxide,ozone,uv_index,temperature_2m,relative_humidity_2m,wind_speed_10m"
)

response = requests.get(url)
data = response.json()

import pprint
pprint.pprint(data)
