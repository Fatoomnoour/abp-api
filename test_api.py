import requests
import json

with open("sample.json", "r") as f:
    data = json.load(f)

res = requests.post("https://web-production-07fc.up.railway.app/predict", json=data)
print(res.json())
