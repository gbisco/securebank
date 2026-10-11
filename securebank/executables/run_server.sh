#!/bin/bash

IMAGE_NAME="securebank"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="$SCRIPT_DIR/../Dockerfile"
BUILD_CONTEXT="$SCRIPT_DIR/../.."

if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$BUILD_CONTEXT"

if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' built successfully."
else
    echo "Error: Failed to build Docker image."
    exit 1
fi

PORT=$(grep -i "EXPOSE" "$DOCKERFILE_PATH" | awk '{print $2}')

if [ -z "$PORT" ]; then
    echo "Error: No port found in Dockerfile."
    exit 1
fi

echo "Running Docker container on port $PORT..."
docker run -d -p "$PORT:$PORT" "$IMAGE_NAME"