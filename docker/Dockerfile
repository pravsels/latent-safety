
# CUDA 12.6.3 + cuDNN
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps, Python 3.12, pip, and create symlink in one layer
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git curl ca-certificates ffmpeg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    # Install python 3.12 and dev
    apt-get install -y \
        python3.12 \
        python3.12-dev && \
    # Install pip manually
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    # Create the symlink
    ln -s /usr/bin/python3.12 /usr/bin/python && \
    # Clean up
    rm -rf /var/lib/apt/lists/*
    
# Prefer PyTorch CUDA 12.4 wheels (to match the base image)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

WORKDIR /workspace

# Install PyTorch, wandb, lerobot and pinned dependencies
RUN set -eux; \
    python -m pip install -v torch==2.6 torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install -v wandb lerobot==0.3.3 "numpy==1.26.4" "gymnasium==0.28.1" torchmetrics timm scipy

# Copy packages (copied to build context by docker_build.sh)
COPY dinov3 /workspace/dinov3
COPY latent_safety /workspace/latent_safety

# Install dinov3
RUN python -m pip install -v -e /workspace/dinov3

# Install latent_safety
RUN python -m pip install -v -e /workspace/latent_safety

CMD ["/bin/bash"]

