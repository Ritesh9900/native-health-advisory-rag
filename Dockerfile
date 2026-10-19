# Production Container for Native Language Health Advisory
FROM python:3.11-slim

# System setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy application files
COPY data/ data/
COPY src/ src/
COPY evals/ evals/
COPY .env.example .env

# Run offline indexing to pre-warm the vector store
RUN python src/rag/ingest.py

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# Production server entrypoint
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
