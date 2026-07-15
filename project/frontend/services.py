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

    El payload coincide con PredictionRequest de backend/models.py. El modelo solo usa el
    texto, el canal y la categoría, así que tipo_cuenta y antigüedad se piden en la interfaz
    pero no se envían al backend. Retorna el JSON de PredictionResponse (clave 'prioridad').
    """
    payload = {
        "asunto_ticket": asunto,
        "contenido_ticket": contenido,
        "canal_ticket": canal_ticket,
        "categoria_problema": categoria_problema,
    }
    respuesta = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=TIMEOUT_SEGUNDOS)
    respuesta.raise_for_status()
    return respuesta.json()
