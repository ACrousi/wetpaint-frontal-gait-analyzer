# ============================================================
# WetPaint Motor Development - Dockerfile
# Base: Python 3.10 (Clean System)
# Includes: PyTorch 2.6.0, CUDA 12.4 support
# ============================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ============================================================
# Install system dependencies
# ============================================================
# build-essential: for compiling python extensions (e.g. cython_bbox)
# git: for git-based dependencies
# ffmpeg, libgl1...: for OpenCV and media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONNOUSERSITE=1

# ============================================================
# Install PyTorch (User Requested)
# ============================================================
# RUN pip install --no-cache-dir \
#     torch==2.6.0 \
#     torchvision==0.21.0 \
#     torchaudio==2.6.0 \
#     --extra-index-url https://download.pytorch.org/whl/cu124

# ============================================================
# GPU available check
# ============================================================
# RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None"

# ============================================================
# Copy vendor packages (Required for requirements.txt)
# ============================================================
# COPY vendor/ /app/vendor/

# ============================================================
# Install Python dependencies
# ============================================================
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Install Vendor Packages (Cached Layer)
# ============================================================
# Copy vendor directory specifically to cache this layer 
# unless vendor packages change.
COPY vendor /app/vendor

# Install vendor packages
# Using --no-deps inside vendor install if their deps are already in requirements.txt
# or let pip handle it.
RUN pip install --no-cache-dir \
    /app/vendor/BoTSORT \
    /app/vendor/rtmlib \
    /app/vendor/ResGCNv1

# ============================================================
# Copy application source code
# ============================================================
COPY . /app

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/models

# ============================================================
# Default command - Run API server
# ============================================================
ENTRYPOINT ["python", "main.py"]
