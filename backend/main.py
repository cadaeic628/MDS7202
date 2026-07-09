from fastapi import FastAPI, HTTPException

from backend.generate_prediction import generate_prediction
from backend.models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="ChaucherApp Ticket Priority API",
    description="API para predecir la prioridad de tickets de soporte al cliente.",
    version="1.0.0",
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend funcionando correctamente"}


@app.post("/predict", response_model=PredictionResponse)
def predict_priority(request: PredictionRequest) -> PredictionResponse:
    try:
        prioridad = generate_prediction(
            asunto_ticket=request.asunto_ticket,
            contenido_ticket=request.contenido_ticket,
            canal_ticket=request.canal_ticket,
            categoria_problema=request.categoria_problema,
        )

        return PredictionResponse(prioridad=prioridad)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        ) from exc
