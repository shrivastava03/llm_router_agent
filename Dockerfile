# --- Builder Stage ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /build

# Install build tools once for all C-based libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Force the correct NumPy version to prevent the "np.float_" crash
RUN pip install --no-cache-dir numpy==1.26.4

# 2. Install ALL necessary dependencies for your LLM Router
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.27.0 \
    tiktoken>=0.7.0 \
    aiosqlite>=0.20.0 \
    tavily-python>=0.3.3 \
    groq>=0.4.0 \
    fastembed>=0.3.0 \
    onnxruntime>=1.18.0 \
    chromadb==0.5.3 \
    pandas>=2.2.0 \
    tabulate>=0.9.0 \
    pymupdf>=1.24.0

# --- Final Stage ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Copy everything from builder to ensure no modules are missing
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your source code
COPY . .

# Final cleanup to keep the size as low as possible
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -delete

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]