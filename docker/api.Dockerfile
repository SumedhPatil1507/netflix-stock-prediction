FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-catalog.txt requirements-streaming.txt ./
RUN pip install --no-cache-dir -r requirements-catalog.txt -r requirements-streaming.txt
COPY api ./api
COPY app ./app
COPY src ./src
COPY config.yaml feature_store.yaml .env.example ./
RUN mkdir -p /app/data/catalog /app/models/recommendations
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
