
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
COPY . .

# Install PyTorch explicitly, then install your package
RUN set -eux; \
    # Install PyTorch==2.6 explicitly using the cu126 index for arm64 + CUDA 12.6
    python -m pip install -v torch==2.6 --index-url https://download.pytorch.org/whl/cu126 && \
    \
    # Install wandb
    python -m pip install -v wandb && \
    \
    # Remove torch requirement from setup.py
    sed -i '/torch/d' setup.py || true; \
    # Install your local package (dependencies except torch)
    python -m pip install -v -e .


CMD ["/bin/bash"]
