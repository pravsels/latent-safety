#!/bin/bash
IMAGE_NAME="latent_safety"

docker run --gpus all --rm -it \
    -v $(pwd)/../:/workspace \
    $IMAGE_NAME
