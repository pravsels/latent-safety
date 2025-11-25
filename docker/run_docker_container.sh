#!/bin/bash
IMAGE_NAME="latent_safety_amd64"

docker run --gpus all --rm -it \
    -v $(pwd)/../:/workspace \
    $IMAGE_NAME
