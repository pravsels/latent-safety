#!/bin/bash
IMAGE_NAME="latent_safety_amd64"

docker run --gpus all --rm -it \
    -e PYTHONPATH=/workspace \
    -v $(pwd)/../:/workspace \
    -v $(pwd)/../../dinov3:/dinov3 \
    $IMAGE_NAME
