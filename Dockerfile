# ============================================================
# WetPaint Motor Development - Dockerfile
# Base: Python 3.10 (Clean System)
# Includes: PyTorch 2.6.0, CUDA 12.4 support
# ============================================================

FROM python:3.10-slim

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
    libglib2.0-0t64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install PyTorch (User Requested)
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================================
# Copy vendor packages (Required for requirements.txt)
# ============================================================
COPY vendor/ /app/vendor/

# ============================================================
# Install Python dependencies
# ============================================================
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Sanity check
# ============================================================
RUN python -c "\
import torch; \
print(f'PyTorch version: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): print(f'CUDA version: {torch.version.cuda}');"

# ============================================================
# Copy application source code
# ============================================================
# COPY src/ /app/src/
# COPY config/ /app/config/
# COPY main.py /app/
# COPY start_api.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/models

# ============================================================
# Environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONNOUSERSITE=1

# ============================================================
# Expose API port
# ============================================================
EXPOSE 8000

# ============================================================
# Default command - Run API server
# ============================================================
CMD ["python", "start_api.py"]
