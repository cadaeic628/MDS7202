FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Instala uv desde la imagen oficial.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Instala dependencias con la versión exacta del lockfile (el venv vive fuera de /app
# para no ser tapado por el volumen montado en docker-compose).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Código de la app.
COPY app.py services.py ./

EXPOSE 7860

CMD ["uv", "run", "--no-sync", "python", "app.py"]
