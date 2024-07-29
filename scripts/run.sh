#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=$3
    local attempt=1

    echo "Waiting for $service_name to start..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null; then
            echo "$service_name is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service_name is not ready yet. Waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "$service_name did not become ready in time."
    return 1
}

# Create a Docker network if it doesn't exist
docker network create achievementsai-network 2>/dev/null || true

echo "Starting Ollama container..."
docker run -d --name ollama-container \
    --network achievementsai-network \
    -p 11434:11434 \
    ollama-image

# Wait for Ollama with a longer timeout (5 minutes)
if ! wait_for_service "Ollama" "http://localhost:11434/api/version" 60; then
    echo "Ollama failed to start in time. Exiting."
    exit 1
fi

echo "Starting Streamlit container..."
docker run -d --name streamlit-container \
    --network achievementsai-network \
    -p 8501:8501 \
    -e OLLAMA_BASE_URL=http://ollama-container:11434 \
    streamlit-image

# Wait for Streamlit with a shorter timeout (1 minute)
if ! wait_for_service "Streamlit" "http://localhost:8501/_stcore/health" 12; then
    echo "Streamlit failed to start in time. Exiting."
    exit 1
fi

echo "Containers are running!"
echo "Streamlit app is available at http://localhost:8501"
echo "Ollama API is available at http://localhost:11434"