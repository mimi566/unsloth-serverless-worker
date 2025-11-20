# ================================
# Base Image: CUDA 12.1 + Ubuntu
# ================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ================================
# System Dependencies
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git wget curl build-essential \
    libssl-dev libsndfile1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ================================
# Install PyTorch (CUDA 12.1 build)
# ================================
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# ================================
# Install Unsloth + Transformers Stack
# ================================
RUN python3 -m pip install \
    unsloth \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    safetensors \
    fastapi \
    uvicorn[standard] \
    sse-starlette

# ================================
# Environment Variables
# ================================
ENV MODEL_NAME="Sourabh66/Llama-2-17B-Fine-Tune-Blog" \
    MAX_SEQ_LENGTH=32768 \
    MODEL_CACHE_DIR="/models" \
    HF_TOKEN="" \
    PYTHONPATH="/app"

# ================================
# Copy App Files
# ================================
WORKDIR /app
COPY src /app/src
COPY server.py /app/server.py
COPY runpod_handler.py /app/runpod_handler.py

# ================================
# (Optional) Model Baking
# ================================
# COPY ./models /models

# ================================
# Expose Port
# ================================
EXPOSE 8080

# ================================
# Start API Server
# ================================
CMD ["python3", "server.py"]
