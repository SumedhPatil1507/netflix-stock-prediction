# ─────────────────────────────────────────────────────────────────────────────
# Alpha Engine — Multi-stage Dockerfile
# Shared image for: api / worker / streamlit services.
# CMD is overridden per-service in docker-compose.yml.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix directory (for copy into final)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Create writable directories
RUN mkdir -p /app/models /app/outputs /app/data /app/logs

# Non-root user for security
RUN adduser --disabled-password --gecos "" alphauser && \
    chown -R alphauser:alphauser /app
USER alphauser

# Health probe port (FastAPI)
EXPOSE 8000
# Streamlit port
EXPOSE 8501

# Default CMD — overridden by docker-compose per service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
