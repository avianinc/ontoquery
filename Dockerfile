# Dockerfile

# Use NVIDIA's CUDA image with Ubuntu 22.04
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV HF_DATASETS_CACHE=/app/cache
ENV HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir faiss-gpu

# Download the SentenceTransformer model during the build
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/cache')"

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
