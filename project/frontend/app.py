# Frontend Gradio de ChaucherApp: formulario para priorizar tickets de soporte.

from __future__ import annotations

import gradio as gr
from services import enviar_prediccion

# Opciones fijas de los atributos categóricos (deben coincidir con las del entrenamiento).
CANALES = ["Correo", "Página Web", "Whatsapp"]
CATEGORIAS = ["Cobros", "Cuenta", "Fraude", "Otro", "Pregunta general", "Técnica"]
TIPOS_CUENTA = ["Business", "Free", "Premium"]


def predecir(asunto, contenido, canal, categoria, tipo_cuenta, antiguedad):
    """Valida el formulario, llama al backend y devuelve el nivel y las probabilidades."""
    if not contenido or not contenido.strip():
        raise gr.Error("El contenido del ticket no puede estar vacío.")
    try:
        data = enviar_prediccion(asunto, contenido, canal, categoria, tipo_cuenta, antiguedad)
    except Exception as exc:  # errores de red o del backend se muestran en la interfaz
        raise gr.Error(f"No se pudo obtener la predicción: {exc}") from exc

    nivel = data.get("nivel_prioridad", "desconocido")
    probabilidades = data.get("probabilidades") or {}
    return nivel, probabilidades


# Tema verde/menta acorde a la estética de una fintech (ChaucherApp = billetera digital).
tema = gr.themes.Soft(primary_hue="emerald", secondary_hue="green", neutral_hue="slate")

with gr.Blocks(theme=tema, title="ChaucherApp · Priorización de tickets") as demo:
    gr.Markdown(
        "# 🪙 ChaucherApp — Priorización de tickets\n"
        "Completa los datos del ticket de soporte y obtén su **nivel de prioridad** "
        "(`Baja`, `Media`, `Alta` o `Critica`) estimado por el modelo."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎫 Datos del ticket")
            asunto = gr.Textbox(label="Asunto", placeholder="Ej: Cobro desconocido en mi tarjeta digital")
            contenido = gr.Textbox(
                label="Contenido",
                lines=7,
                placeholder="Describe el problema tal como lo escribiría el usuario...",
            )
            canal = gr.Dropdown(CANALES, label="Canal del ticket", value="Whatsapp")
            categoria = gr.Dropdown(CATEGORIAS, label="Categoría del problema", value="Cuenta")

        with gr.Column():
            gr.Markdown("## 👤 Datos del usuario")
            tipo_cuenta = gr.Dropdown(TIPOS_CUENTA, label="Tipo de cuenta", value="Free")
            antiguedad = gr.Slider(
                minimum=0,
                maximum=1200,
                value=120,
                step=1,
                label="Antigüedad de la cuenta (días)",
            )
            gr.Markdown(
                "El modelo prioriza según el contenido del ticket y sus atributos. "
                "Los casos críticos (por ejemplo fraude) deberían clasificarse con mayor prioridad."
            )

    boton = gr.Button("Predecir prioridad", variant="primary")

    with gr.Row():
        salida_nivel = gr.Label(label="Nivel de prioridad")
        salida_probabilidades = gr.Label(label="Probabilidades por clase", num_top_classes=4)

    boton.click(
        predecir,
        inputs=[asunto, contenido, canal, categoria, tipo_cuenta, antiguedad],
        outputs=[salida_nivel, salida_probabilidades],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
