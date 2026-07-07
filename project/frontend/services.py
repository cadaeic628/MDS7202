# Cliente HTTP del frontend: llama al endpoint /predict del backend.

from __future__ import annotations

import os

import requests

# URL del backend. En docker-compose se define como http://backend:8000; en local, http://localhost:8000.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TIMEOUT_SEGUNDOS = 30


def enviar_prediccion(
    asunto: str,
    contenido: str,
    canal_ticket: str,
    categoria_problema: str,
    tipo_cuenta: str,
    antiguedad_cuenta_dias: int,
) -> dict:
    """Envía los datos del ticket al backend y retorna la respuesta del modelo.

    El esquema del payload debe coincidir con PredictionRequest de backend/models.py.
    Retorna el JSON de PredictionResponse (al menos con la clave 'nivel_prioridad').
    """
    payload = {
        "asunto": asunto,
        "contenido": contenido,
        "canal_ticket": canal_ticket,
        "categoria_problema": categoria_problema,
        "tipo_cuenta": tipo_cuenta,
        "antiguedad_cuenta_dias": int(antiguedad_cuenta_dias),
    }
    respuesta = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()
    return respuesta.json()
