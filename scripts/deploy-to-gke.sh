#!/bin/bash

# Variables (Replace with your values)
PROJECT_ID="achievementsai"
REGION="us-central1" # Change to your preferred region
CLUSTER_NAME="my-cluster"
ZONE="us-central1-a" # Change to your preferred zone
IMAGE_TAG="latest"

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
gcloud auth login

# Set the project
gcloud config set project $PROJECT_ID

# Create a GKE Cluster
echo "Creating GKE cluster..."
gcloud container clusters create $CLUSTER_NAME \
  --zone $ZONE \
  --num-nodes=1

# Configure kubectl to use the GKE cluster
echo "Configuring kubectl..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

# Build Docker images
echo "Building Docker images..."
docker build -f Dockerfile.ollama -t gcr.io/$PROJECT_ID/ollama-image:$IMAGE_TAG .
docker build -f Dockerfile.streamlit -t gcr.io/$PROJECT_ID/streamlit-image:$IMAGE_TAG .

# Authenticate Docker with Google Container Registry
echo "Authenticating Docker with Google Container Registry..."
gcloud auth configure-docker

# Push Docker images to Google Container Registry
echo "Pushing Docker images to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/ollama-image:$IMAGE_TAG
docker push gcr.io/$PROJECT_ID/streamlit-image:$IMAGE_TAG

# Convert Docker Compose to Kubernetes manifests
echo "Converting Docker Compose to Kubernetes manifests..."
kompose convert -f docker-compose.yml

# Deploy to GKE
echo "Deploying to GKE..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Optionally, create LoadBalancer services for external access
# Uncomment and modify if necessary
# echo "Creating LoadBalancer services..."
# kubectl apply -f loadbalancer-service.yaml

# Verify the deployment
echo "Verifying the deployment..."
kubectl get pods
kubectl get services

echo "Deployment complete."

# Output service URLs (modify based on your setup)
echo "You can access your services via the following URLs:"
echo "Ollama Service URL: $(kubectl get svc ollama-service --output jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
echo "Streamlit Service URL: $(kubectl get svc streamlit-service --output jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
