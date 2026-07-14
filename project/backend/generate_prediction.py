import os
from pathlib import Path

import cloudpickle
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / "project" / ".env"
MODEL_PATH = BASE_DIR / "modelo_final.pkl"

load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def load_pipeline(model_path: Path = MODEL_PATH):
    """Carga el pipeline entrenado en la Parte 2."""
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    with open(model_path, "rb") as file:
        pipeline = cloudpickle.load(file)

    return pipeline


PIPELINE = load_pipeline()


def get_used_columns_from_pipeline(pipeline) -> list[str]:
    """Obtiene las columnas originales que espera el ColumnTransformer."""
    column_transformer = pipeline.named_steps["Preprocessing"]

    used_columns = []

    for _, transformer, columns in column_transformer.transformers:
        if transformer == "drop":
            continue

        if isinstance(columns, str):
            used_columns.append(columns)
        else:
            used_columns.extend(list(columns))

    return used_columns


USED_COLUMNS = get_used_columns_from_pipeline(PIPELINE)


def build_text_for_embedding(asunto_ticket: str, contenido_ticket: str) -> str:
    """Construye el texto que se enviará al modelo de embeddings."""
    return f"Asunto_Ticket: {asunto_ticket}\nContenido_Ticket: {contenido_ticket}"


def build_text_for_model(asunto_ticket: str, contenido_ticket: str) -> str:
    """Construye la columna Texto igual que en la Parte 2."""
    return f"{asunto_ticket} {contenido_ticket}"


def generate_embedding(texto: str) -> list[float]:
    """Genera un embedding de 1024 dimensiones con Gemini."""
    if GOOGLE_API_KEY is None:
        raise ValueError(f"No se encontró GOOGLE_API_KEY. Revisa que exista el archivo: {ENV_PATH}")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1024,
    )

    embedding = embedding_model.embed_query(texto)

    if len(embedding) != 1024:
        raise ValueError(f"El embedding debería tener 1024 dimensiones, pero tiene {len(embedding)}.")

    return embedding


def build_input_dataframe(
    asunto_ticket: str,
    contenido_ticket: str,
    canal_ticket: str,
    categoria_problema: str,
) -> pd.DataFrame:
    """Construye el DataFrame de entrada con las columnas esperadas por el modelo."""
    texto_embedding = build_text_for_embedding(
        asunto_ticket=asunto_ticket,
        contenido_ticket=contenido_ticket,
    )

    texto_modelo = build_text_for_model(
        asunto_ticket=asunto_ticket,
        contenido_ticket=contenido_ticket,
    )

    embedding = generate_embedding(texto_embedding)

    row = {
        "N_Caracteres_Ticket": len(texto_modelo),
        "Canal_Ticket": canal_ticket,
        "Categoría_Problema": categoria_problema,
        "Texto": texto_modelo,
    }

    embedding_columns = [column for column in USED_COLUMNS if column not in row]

    if len(embedding_columns) > 0:
        if len(embedding_columns) != len(embedding):
            raise ValueError(
                f"El modelo espera {len(embedding_columns)} columnas de embedding, "
                f"pero se generaron {len(embedding)} dimensiones."
            )

        for column, value in zip(embedding_columns, embedding, strict=False):
            row[column] = value

    input_df = pd.DataFrame([row])

    return input_df


def generate_prediction(
    asunto_ticket: str,
    contenido_ticket: str,
    canal_ticket: str,
    categoria_problema: str,
) -> str:
    """Genera la predicción de prioridad para un ticket."""
    input_df = build_input_dataframe(
        asunto_ticket=asunto_ticket,
        contenido_ticket=contenido_ticket,
        canal_ticket=canal_ticket,
        categoria_problema=categoria_problema,
    )

    prediction = PIPELINE.predict(input_df)[0]

    return str(prediction)


if __name__ == "__main__":
    prediccion = generate_prediction(
        asunto_ticket="No puedo acceder a mi cuenta",
        contenido_ticket=(
            "Hola, desde ayer no puedo iniciar sesión en mi cuenta. "
            "Intenté recuperar la contraseña, pero no recibo el correo de recuperación."
        ),
        canal_ticket="Correo",
        categoria_problema="Cuenta",
    )

    print("Predicción generada:")
    print(prediccion)
