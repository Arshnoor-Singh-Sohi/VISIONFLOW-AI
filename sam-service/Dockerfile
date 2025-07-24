# =============================================================================
# VisionFlow AI - SAM Service Dockerfile
# =============================================================================
# This Dockerfile creates a container for running the Segment Anything Model (SAM)
# It includes all necessary dependencies and optimizations for both CPU and GPU usage

# Use PyTorch base image with CUDA support (can fall back to CPU)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    # Image processing libraries
    libopencv-dev \
    python3-opencv \
    # System utilities
    curl \
    wget \
    unzip \
    # For monitoring and debugging
    htop \
    nvtop \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create necessary directories
RUN mkdir -p /app/models \
             /app/temp \
             /app/segments \
             /app/logs

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM from source (latest version)
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Copy application code
COPY app.py .
COPY sam_processor.py .
COPY utils.py .

# Create a non-root user for security
RUN groupadd -r samuser && useradd -r -g samuser samuser

# Set proper permissions
RUN chown -R samuser:samuser /app
USER samuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "app.py"]

# =============================================================================
# Build and run instructions:
# 
# Build the image:
#   docker build -t visionflow-sam .
#
# Run with CPU:
#   docker run -p 8001:8000 -v ./data/models:/app/models visionflow-sam
#
# Run with GPU:
#   docker run --gpus all -p 8001:8000 -v ./data/models:/app/models visionflow-sam
#
# Environment variables you can set:
#   -e DEVICE=cuda          # Use GPU if available
#   -e MODEL_PATH=/app/models/sam_vit_h_4b8939.pth
#   -e MAX_IMAGE_SIZE=2048
#   -e CLEANUP_INTERVAL=3600
# =============================================================================