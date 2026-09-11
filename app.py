from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Cargar modelo
model = joblib.load("model/cliente_purchase_model.pkl")

# Crear API
app = FastAPI(
    title="Customer Purchase Prediction API",
    description="API para predecir la probabilidad de compra de clientes",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def web_app():
    return FileResponse("static/index.html")

# Datos de entrada
class CustomerFeatures(BaseModel):
    compras_30d: int
    gasto_30d: float
    compras_60d: int
    gasto_60d: float
    ticket_medio_60d: float
    dias_desde_ultima_compra: int


@app.get("/")
def root():
    return {
        "message": "Customer Purchase Prediction API",
        "status": "online"
    }


@app.post("/predict")
def predict(features: CustomerFeatures):

    # Convertir entrada a DataFrame
    data = pd.DataFrame([features.model_dump()])

    # Predicción
    prediction = model.predict(data)[0]

    # Probabilidad de compra
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }