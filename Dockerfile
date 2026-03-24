# ─────────────────────────────────────────────────────────────────────────────
# Research Agent — Dockerfile
# Build:   docker build -t research-agent .
# Run:     docker run -p 8501:8501 --env-file .env research-agent
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: dependency builder (keeps final image lean)
FROM python:3.11-slim AS builder

WORKDIR /install

# System deps needed to build PyMuPDF + FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install/pkg --no-cache-dir -r requirements.txt


# Stage 2: final slim runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="your-email@example.com"
LABEL description="Agentic Research Paper Assistant — 100% open-source"

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/pkg /usr/local

# System runtime libs only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY src/ ./src/

# Pre-download the embedding model so first run is instant
# (model is cached in /app/.cache — volume-mount this for faster restarts)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Switch to non-root user
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
