#!/bin/bash

# Configuration Variables
PROJECT_ID="achievementsai"
IMAGE_NAME="achievementsai-image"
TAG="latest"
REGION="us-central1"
SERVICE_NAME="achievementsai"
ENV_FILE=".env"

# Ensure the script exits on error
set -e

# Check if gcloud and docker are installed
command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required but it's not installed. Exiting."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker is required but it's not installed. Exiting."; exit 1; }

# Authenticate gcloud (optional if already authenticated)
echo "Authenticating with Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Build the Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG .

# Push the Docker image to Google Container Registry
echo "Pushing Docker image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG

# Function to load environment variables from .env file
load_env_variables() {
    if [[ -f $ENV_FILE ]]; then
        export $(grep -v '^#' $ENV_FILE | xargs)
    fi
}

# Load environment variables
load_env_variables

# Construct the --set-env-vars flag
ENV_VARS=$(grep -v '^#' $ENV_FILE | xargs | sed 's/ /,/g')

# Deploy the Docker image to Google Cloud Run with environment variables
echo "Deploying Docker image to Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars $ENV_VARS

echo "Deployment complete. Service URL:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)'
