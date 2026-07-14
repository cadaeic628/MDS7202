from pydantic import BaseModel, Field, StrictStr


class PredictionRequest(BaseModel):
    asunto_ticket: StrictStr = Field(
        ...,
        description="Asunto o título del ticket.",
        examples=["No puedo acceder a mi cuenta"],
    )
    contenido_ticket: StrictStr = Field(
        ...,
        description="Descripción completa del problema reportado por el usuario.",
        examples=[
            "Hola, desde ayer no puedo iniciar sesión. Intenté recuperar la contraseña, pero no recibo el correo."
        ],
    )
    canal_ticket: StrictStr = Field(
        ...,
        description="Canal por el que ingresó el ticket.",
        examples=["Correo"],
    )
    categoria_problema: StrictStr = Field(
        ...,
        description="Categoría del problema reportado.",
        examples=["Cuenta"],
    )


class PredictionResponse(BaseModel):
    prioridad: StrictStr = Field(
        ...,
        description="Nivel de prioridad predicho por el modelo.",
        examples=["Media"],
    )
