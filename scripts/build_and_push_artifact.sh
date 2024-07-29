#!/bin/bash

# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"  # Replace with your desired region
export REPOSITORY="achievementsai-repo"  # Replace with your repository name

# Build the Docker image
echo "Building Docker image..."
docker build -t achievementsai-gpu .

# Tag the image for Artifact Registry
echo "Tagging image for Artifact Registry..."
docker tag achievementsai-gpu ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/achievementsai-gpu:latest

# Configure Docker to use gcloud as a credential helper for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Push the image to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/achievementsai-gpu:latest

echo "Image pushed successfully to Artifact Registry!"