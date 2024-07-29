#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Building Ollama container..."
docker build -t ollama-image -f Dockerfile.ollama .

echo "Building Streamlit container..."
docker build -t streamlit-image -f Dockerfile.streamlit .

echo "Build completed successfully!"