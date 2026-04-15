# Use an official, lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install C++ compilers so ChromaDB builds perfectly without hanging
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip to prevent dependency resolution errors
RUN pip install --upgrade pip

# 1. Install CPU-only PyTorch FIRST
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install Vector DB
RUN pip install chromadb==0.5.0

# 3. Install Embedding Library
RUN pip install sentence-transformers>=2.7.0

# 4. Install the rest of the lightweight dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace AI model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download the ChromaDB ONNX model
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['test'])"

# Copy the rest of your application code
COPY . .

# Expose the default port
EXPOSE 10000

# Bulletproof start command natively evaluated by Docker for Render
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}