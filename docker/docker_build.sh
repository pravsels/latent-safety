#!/bin/bash

# Select platform: set to either "arm64" or "amd64"
PLATFORM="arm64"

case "$PLATFORM" in
  arm64|amd64)
    ;;
  *)
    echo "Unsupported PLATFORM '$PLATFORM'. Use 'arm64' or 'amd64'."
    exit 1
    ;;
esac

IMAGE_NAME="latent_safety_${PLATFORM}"

# docker buildx create --use
docker buildx build --platform "linux/${PLATFORM}" -t "$IMAGE_NAME" --load -f Dockerfile ..
