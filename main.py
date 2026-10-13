from fastapi import FastAPI
import os
import requests

app = FastAPI(title="MLOps Databricks API")

DATABRICKS_ENDPOINT = os.getenv("DATABRICKS_ENDPOINT_URL", "https://adb-xxxx.azuredatabricks.net/serving-endpoints/tu-modelo/invocations")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "token-final")

@app.get("/")
def read_root():
    return {"message": "API de MLOps funcionando correctamente y conectada a Databricks"}

@app.post("/predict")
def predict(data: dict):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(DATABRICKS_ENDPOINT, json=data, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
