#!/bin/bash
IMAGE_NAME="latent_safety"
 
docker build -t $IMAGE_NAME -f Dockerfile .. 

# docker buildx create --use
# docker buildx build --platform linux/arm64 -t latent_safety -f Dockerfile .. 
# docker buildx build --platform linux/arm64 -t latent_safety --load -f Dockerfile ..
