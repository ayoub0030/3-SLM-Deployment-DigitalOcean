#!/bin/bash

echo "Deploying 3-SLM services to DigitalOcean..."

# Build and deploy with Docker Compose
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo "Checking service health..."
sleep 30

echo "Testing endpoints..."
curl -X GET http://localhost/qwen/health
curl -X GET http://localhost/phi2/health  
curl -X GET http://localhost/gemma/health

echo "Deployment complete!"
echo "Access your services at:"
echo "- Qwen: http://your-droplet-ip/qwen/"
echo "- Phi-2: http://your-droplet-ip/phi2/"
echo "- Gemma: http://your-droplet-ip/gemma/"