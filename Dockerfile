# Use an official, lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# 🔥 THE FIX: Install CPU-only PyTorch FIRST. (Shrinks download from 2.5GB to 150MB)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Now install the rest of the requirements
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Pre-download the embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]