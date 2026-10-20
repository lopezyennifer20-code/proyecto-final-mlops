import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API Telco Churn")

DATABRICKS_URL = os.environ["DATABRICKS_URL"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]


class Entrada(BaseModel):
    gender: float
    SeniorCitizen: float
    Partner: float
    Dependents: float
    tenure: float
    PhoneService: float
    MultipleLines: float
    InternetService: float
    OnlineSecurity: float
    OnlineBackup: float
    DeviceProtection: float
    TechSupport: float
    StreamingTV: float
    StreamingMovies: float
    Contract: float
    PaperlessBilling: float
    PaymentMethod: float
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predecir")
def predecir(datos: Entrada):
    payload = {"dataframe_records": [datos.model_dump()]}
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    r = requests.post(DATABRICKS_URL, json=payload, headers=headers, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    return {"predictions": r.json()["predictions"]}
