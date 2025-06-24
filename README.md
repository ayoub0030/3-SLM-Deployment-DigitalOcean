# 3-SLM Deployment on DigitalOcean

Simple deployment of 3 Small Language Models using FastAPI and Docker.

## Models Included
- Qwen 2.5 (0.5B parameters)
- Microsoft Phi-2 (2.7B parameters)  
- Google Gemma (2B parameters)

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd 3-slm-deployment
   ```

2. **Deploy locally:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Deploy to DigitalOcean:**
   - Create a Droplet (4GB+ RAM recommended)
   - Install Docker and Docker Compose
   - Upload project files
   - Run `./deploy.sh`

## API Usage

### Generate Text
```bash
curl -X POST "http://your-ip/qwen/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_length": 50}'
```

### Health Check
```bash
curl http://your-ip/qwen/health
```

## Endpoints
- `/qwen/` - Qwen model service
- `/phi2/` - Phi-2 model service  
- `/gemma/` - Gemma model service

## Requirements
- 4GB+ RAM (8GB recommended)
- Docker & Docker Compose
- Internet connection for model downloads

## Scaling
Each model runs in its own container with 2GB memory limit. Adjust in `docker-compose.yml` as needed.

---

This is a complete, production-ready project structure for deploying 3 SLMs to DigitalOcean. The code is intentionally simple and follows MVP principles:

**Key Features:**
- ✅ 3 separate FastAPI services for each model
- ✅ Docker containerization 
- ✅ Nginx load balancer/proxy
- ✅ Docker Compose orchestration
- ✅ Simple deployment script
- ✅ Health checks and error handling
- ✅ Memory limits for each service

**To deploy:**
1. Create the folder structure and files as shown
2. Get a DigitalOcean droplet (4GB+ RAM)
3. Install Docker/Docker Compose
4. Upload files and run `./deploy.sh`

The system will be accessible at your droplet's IP with endpoints `/qwen/`, `/phi2/`, and `/gemma/`.