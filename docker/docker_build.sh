#!/bin/bash
set -e

# Select platform: set to either "arm64" or "amd64"
PLATFORM="${1:-amd64}"
IMAGE_NAME="latent_safety_${PLATFORM}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DINOV3_DIR="$(dirname "$REPO_DIR")/dinov3"

[ -d "$DINOV3_DIR" ] || { echo "dinov3 repo not found at $DINOV3_DIR"; exit 1; }

BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Copy all package directories
cp -r "$REPO_DIR" "$BUILD_DIR/latent_safety"
cp -r "$DINOV3_DIR" "$BUILD_DIR/dinov3"
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/"
[ -f "$SCRIPT_DIR/.dockerignore" ] && cp "$SCRIPT_DIR/.dockerignore" "$BUILD_DIR/"

if [ "$PLATFORM" = "arm64" ]; then
    docker buildx build \
        --platform "linux/${PLATFORM}" \
        -t "$IMAGE_NAME" \
        --load \
        -f "$BUILD_DIR/Dockerfile" \
        "$BUILD_DIR"
else
    docker build \
        -t "$IMAGE_NAME" \
        -f "$BUILD_DIR/Dockerfile" \
        "$BUILD_DIR"
fi

echo "Built: $IMAGE_NAME"
