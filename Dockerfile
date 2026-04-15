# ============================================================================
# Stage 1: Builder (compile heavy deps)
# ============================================================================
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install system build tools AND libgomp1 so ONNX verification doesn't fail
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

RUN pip install --upgrade --no-cache-dir pip setuptools wheel

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    sentence-transformers>=2.7.0 \
    chromadb==0.5.0 \
    onnxruntime

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt

# Verify critical imports
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import sentence_transformers; print(f'Sentence Transformers OK')" && \
    python -c "import chromadb; print(f'ChromaDB version: {chromadb.__version__}')" && \
    python -c "import onnxruntime; print(f'ONNX Runtime OK')"

# ============================================================================
# Stage 2: Runtime (slim, only what's needed)
# ============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 🔥 HOTFIX 1: Add libgomp1 to the runtime stage so ONNX doesn't crash!
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy pip packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Cache directories setup
RUN mkdir -p /app/.cache/sentence-transformers \
    && mkdir -p /app/.cache/huggingface \
    && chmod -R 777 /app/.cache

# 🔥 HOTFIX 2: Pre-download both models INTO the newly created cache folders
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['test'])"

# Copy app code LAST
COPY . .

# Verify app structure exists
RUN test -f api/main.py || (echo "ERROR: api/main.py not found" && exit 1) && \
    test -f requirements.txt || (echo "ERROR: requirements.txt not found" && exit 1)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]