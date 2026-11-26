# API image for `dr-referable` (Phase 6). Built by `docker compose up -d api`.
# CPU-only TensorFlow; the model itself is pulled from the MLflow registry at startup, never baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs && chown app /app/logs
USER app

EXPOSE 8000
CMD ["python", "-m", "dr.api", "--host", "0.0.0.0", "--port", "8000"]
