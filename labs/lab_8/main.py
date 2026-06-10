# Sirve el clasificador de potabilidad entrenado en optimize.py vía FastAPI.

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

FEATURE_COLS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]


class WaterSample(BaseModel):
    ph: float = Field(..., description="pH del agua")
    Hardness: float = Field(..., description="Dureza")
    Solids: float = Field(..., description="Sólidos disueltos totales (TDS)")
    Chloramines: float = Field(..., description="Cloraminas")
    Sulfate: float = Field(..., description="Sulfato")
    Conductivity: float = Field(..., description="Conductividad")
    Organic_carbon: float = Field(..., description="Carbono orgánico")
    Trihalomethanes: float = Field(..., description="Trihalometanos")
    Turbidity: float = Field(..., description="Turbidez")


class PotabilityResponse(BaseModel):
    potabilidad: int


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    with open(MODEL_PATH, "rb") as file:
        state["model"] = pickle.load(file)
    yield
    state.clear()


app = FastAPI(
    title="Clasificador de Potabilidad del Agua",
    description="API REST que predice si una muestra de agua es potable usando un XGBoost optimizado con Optuna.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict:
    return {
        "modelo": "XGBoost (con imputación por mediana) optimizado con Optuna y registrado en MLflow.",
        "problema": (
            "Clasificación binaria de potabilidad del agua a partir de 9 mediciones químicas obtenidas "
            "desde sensores IOT, para alertar de forma inmediata cuando el agua no es apta para consumo humano."
        ),
        "entrada": {col: "float" for col in FEATURE_COLS},
        "salida": {"potabilidad": "0 = no potable, 1 = potable"},
        "endpoints": {
            "GET /": "Descripción del servicio.",
            "POST /potabilidad/": "Predice la potabilidad de una muestra de agua.",
        },
    }


@app.post("/potabilidad/", response_model=PotabilityResponse)
def predict_potability(sample: WaterSample) -> PotabilityResponse:
    df = pd.DataFrame([sample.model_dump()], columns=FEATURE_COLS)
    prediction = int(state["model"].predict(df)[0])
    return PotabilityResponse(potabilidad=prediction)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
