# ============================================================
# WetPaint Motor Development - Dockerfile
# Base: NGC Optimized PyTorch 24.05
# Includes: PyTorch 2.4, CUDA 12.4, cuDNN 9, Python 3.10
# ============================================================

FROM nvcr.io/nvidia/pytorch-24.11-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# ============================================================
# Install system dependencies (FFmpeg and OpenCV requirements)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Copy vendor packages first (for better Docker layer caching)
# ============================================================
COPY vendor/ /app/vendor/

# ============================================================
# Install Python dependencies
# ============================================================
# Create constraints file to prevent overwriting NGC's PyTorch
RUN pip freeze | grep -E "^torch" > /tmp/constraints.txt || true

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

# ============================================================
# Sanity check - verify PyTorch and CUDA are working correctly
# ============================================================
RUN python -c "\
import torch; \
print(f'PyTorch version: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA version: {torch.version.cuda}'); \
assert torch.cuda.is_available(), 'CUDA not available!'; \
print('✅ PyTorch + CUDA sanity check passed!')"

# ============================================================
# Copy application source code
# ============================================================
COPY src/ /app/src/
COPY config/ /app/config/
COPY main.py /app/
COPY start_api.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/models

# ============================================================
# Environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ============================================================
# Expose API port
# ============================================================
EXPOSE 8000

# ============================================================
# Default command - Run API server
# ============================================================
CMD ["python", "start_api.py"]
