# PII Security Module — runtime image.
# The trained GLiNER checkpoint is copied into the image so workers can load it
# locally without network access.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for torch/transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application source.
COPY src ./src
COPY ru_gliner_hybrid_v4/ru_pii ./ru_gliner_hybrid_v4/ru_pii
COPY ru_gliner_hybrid_v4/models/gliner-ru-pii-small ./ru_gliner_hybrid_v4/models/gliner-ru-pii-small
# Distilled student checkpoint (fast detector).
COPY artifacts/student-pii.pt ./artifacts/student-pii.pt

ENV PYTHONPATH=/app:/app/ru_gliner_hybrid_v4

# Default: run the HTTP API. Override CMD to run workers (see docker-compose).
EXPOSE 8000
CMD ["python", "-m", "src.run_api"]