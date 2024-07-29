#!/bin/bash

# Variables (Replace with your values)
PROJECT_ID="achievementsai"
REGION="us-central1" # Change to your preferred region
IMAGE_TAG="latest"

# Build Docker images
echo "Building Docker images..."

docker build -f Dockerfile.ollama -t gcr.io/$PROJECT_ID/ollama-image:$IMAGE_TAG .
docker build -f Dockerfile.streamlit -t gcr.io/$PROJECT_ID/streamlit-image:$IMAGE_TAG .

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
gcloud auth configure-docker

# Push Docker images to Google Container Registry
echo "Pushing Docker images to Google Container Registry..."

docker push gcr.io/$PROJECT_ID/ollama-image:$IMAGE_TAG
docker push gcr.io/$PROJECT_ID/streamlit-image:$IMAGE_TAG

# Deploy to Cloud Run
echo "Deploying Ollama to Google Cloud Run..."
gcloud run deploy ollama \
  --image gcr.io/$PROJECT_ID/ollama-image:$IMAGE_TAG \
  --platform managed \
  --region $REGION \
  --port 11434 \
  --allow-unauthenticated

# Extract the URL of the deployed Ollama service
OLLAMA_URL=$(gcloud run services describe ollama --platform managed --region $REGION --format "value(status.url)")

# Deploy Streamlit to Cloud Run
echo "Deploying Streamlit to Google Cloud Run..."
gcloud run deploy streamlit \
  --image gcr.io/$PROJECT_ID/streamlit-image:$IMAGE_TAG \
  --platform managed \
  --region $REGION \
  --port 8501 \
  --allow-unauthenticated \
  --set-env-vars OLLAMA_BASE_URL=$OLLAMA_URL

echo "Deployment complete."

# Output the URLs
echo "Ollama URL: $OLLAMA_URL"
echo "Streamlit URL: $(gcloud run services describe streamlit --platform managed --region $REGION --format "value(status.url)")"
