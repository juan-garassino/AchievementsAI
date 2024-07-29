#!/bin/bash

# Set project ID
export PROJECT_ID=$(gcloud config get-value project)

# Build the Docker image
docker build --no-cache -t achievementsai-gpu . #--no-cache 

# Tag the image for GCR
docker tag achievementsai-gpu gcr.io/$PROJECT_ID/achievementsai-gpu:latest

# Push the image to GCR
docker push gcr.io/$PROJECT_ID/achievementsai-gpu:latest