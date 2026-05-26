FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/service-api/service-python:/app/client-cli

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY service-api/service-python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY service-api ./service-api
COPY client-cli ./client-cli
COPY scripts ./scripts
COPY docs ./docs
COPY README.md LICENSE ./

RUN mkdir -p /app/runtime/data /app/runtime/logs /app/runtime/exports /app/runtime/trained_models /app/runtime/mlruns

CMD ["python", "client-cli/main.py"]
