#!/bin/bash
set -e

# ============================================================
# WetPaint Motor Development - Environment Setup Script
# Replicates the behavior of the Dockerfile
# Base Image Requirement: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# ============================================================

echo "Starting environment setup..."

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or use sudo to install system dependencies."
  exit 1
fi

# ============================================================
# 1. Install system dependencies (Matches Dockerfile lines 16-28)
# ============================================================
echo "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Set Environment Variables (Matches Dockerfile lines 31-35)
# ============================================================
# Note: These are exported for the duration of this script.
# For persistent usage, you might need to add them to ~/.bashrc or similar.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

# ============================================================
# 3. Install Python dependencies (Matches Dockerfile lines 57-60)
# ============================================================
echo "Installing Python dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# ============================================================
# 4. Install Vendor Packages (Matches Dockerfile lines 62-75)
# ============================================================
echo "Installing vendor packages..."
pip install --no-cache-dir \
    ./vendor/BoTSORT \
    ./vendor/rtmlib \
    ./vendor/ResGCNv1

# ============================================================
# 5. Create necessary directories (Matches Dockerfile line 83)
# ============================================================
echo "Creating application directories..."
mkdir -p data outputs logs models

echo "Environment setup complete! You are ready to run."
echo "To start the application, run: python main.py"
