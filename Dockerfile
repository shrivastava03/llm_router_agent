# Use an official, lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install CPU-only PyTorch FIRST to save RAM and time
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 🔥 Pre-download the SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 🔥 THE NEW FIX: Pre-download the ChromaDB ONNX memory model
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['test'])"

# Copy the rest of the application code
COPY . .

# Expose a default port
EXPOSE 10000

# Let the Docker shell evaluate Render's dynamic port natively
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}