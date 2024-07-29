#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables - Replace with your actual values
PROJECT_ID="achievementsai"
CLUSTER_NAME="achievementsai"
ZONE="us-central1-a"
OLLAMA_IMAGE="ollama/ollama"
STREAMLIT_IMAGE="achievementsai-streamlit-app"
OLLAMA_TAG="gcr.io/$PROJECT_ID/ollama:latest"
STREAMLIT_TAG="gcr.io/$PROJECT_ID/streamlit:latest"

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
gcloud auth login

# Set the project
echo "Setting the project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable necessary services
echo "Enabling necessary APIs..."
gcloud services enable container.googleapis.com containerregistry.googleapis.com

# Create a Kubernetes cluster
echo "Creating a Kubernetes cluster..."
gcloud container clusters create $CLUSTER_NAME --zone $ZONE

# Get credentials for kubectl
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

# Tag and push Docker images to Google Container Registry
echo "Tagging and pushing Docker images to Google Container Registry..."
docker tag $OLLAMA_IMAGE $OLLAMA_TAG
docker tag $STREAMLIT_IMAGE $STREAMLIT_TAG
docker push $OLLAMA_TAG
docker push $STREAMLIT_TAG

# Install kompose if not installed
if ! [ -x "$(command -v kompose)" ]; then
  echo "Installing kompose..."
  curl -L https://github.com/kubernetes/kompose/releases/download/v1.21.0/kompose-linux-amd64 -o kompose
  chmod +x kompose
  sudo mv kompose /usr/local/bin/kompose
fi

# Convert Docker Compose to Kubernetes manifests
echo "Converting Docker Compose to Kubernetes manifests..."
kompose convert -f docker-compose.yml

# Modify the Ollama service to expose port 11434
echo "Modifying Ollama service to expose port 11434..."
sed -i 's/port: 11434/port: 11434\n    targetPort: 11434/' ollama-container-service.yaml

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f ollama-container-service.yaml
kubectl apply -f ollama-container-deployment.yaml
kubectl apply -f streamlit-app-service.yaml
kubectl apply -f streamlit-app-deployment.yaml

# Expose the Ollama service within the cluster
echo "Exposing Ollama service within the cluster..."
kubectl expose deployment ollama-container --port=11434 --target-port=11434

# Expose the Streamlit app via an external load balancer
echo "Exposing the Streamlit app via an external load balancer..."
kubectl expose deployment streamlit-app --type=LoadBalancer --port=8501 --target-port=8501

# Verify the deployment
echo "Verifying the deployment..."
kubectl get services
kubectl get deployments

echo "Deployment to GCP using GKE completed successfully."

# Print information about accessing the services
echo "Ollama service is accessible within the cluster at: ollama-container:11434"
echo "Waiting for Streamlit external IP..."
external_ip=""
while [ -z $external_ip ]; do
  echo "Waiting for end point..."
  external_ip=$(kubectl get svc streamlit-app --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}")
  [ -z "$external_ip" ] && sleep 10
done
echo "Streamlit app is accessible at: http://$external_ip:8501"
