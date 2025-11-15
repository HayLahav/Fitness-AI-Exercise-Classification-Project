# Deployment Guide

## Docker Deployment (Recommended)

### Prerequisites
- Docker installed
- Docker Compose installed (optional but recommended)

### Using Docker Compose

1. **Build and start the service:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f
```

3. **Stop the service:**
```bash
docker-compose down
```

### Using Docker Directly

1. **Build the image:**
```bash
docker build -t fitness-ai:latest .
```

2. **Run the container:**
```bash
docker run -d \
  --name fitness-ai-api \
  -p 8000:8000 \
  -v $(pwd)/Saved_model:/app/Saved_model:ro \
  fitness-ai:latest
```

3. **Check logs:**
```bash
docker logs -f fitness-ai-api
```

4. **Stop the container:**
```bash
docker stop fitness-ai-api
docker rm fitness-ai-api
```

## Manual Deployment

### Development Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start development server with auto-reload
uvicorn fitness_ai.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn + Uvicorn workers
gunicorn fitness_ai.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

## Cloud Deployment

### AWS ECS/Fargate

1. Push Docker image to ECR
2. Create task definition using the image
3. Create ECS service
4. Configure load balancer

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/fitness-ai

# Deploy
gcloud run deploy fitness-ai \
  --image gcr.io/PROJECT_ID/fitness-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku

1. Create `Procfile`:
```
web: uvicorn fitness_ai.api.main:app --host 0.0.0.0 --port $PORT
```

2. Deploy:
```bash
heroku create fitness-ai
heroku container:push web
heroku container:release web
```

## Environment Variables

```bash
# Model paths
MODEL_SAVE_PATH=/path/to/models
DATASET_PATH=/path/to/data

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/fitness_ai.log

# API settings
API_HOST=0.0.0.0
API_PORT=8000
```

## Health Monitoring

The API includes a health check endpoint at `/health`:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## Performance Tuning

### Gunicorn Workers

Calculate optimal workers: `(2 x CPU_COUNT) + 1`

```bash
# For 4 CPUs
gunicorn ... --workers 9
```

### Request Timeouts

For video processing, increase timeout:

```bash
gunicorn ... --timeout 300  # 5 minutes
```

## Security Best Practices

1. **Use HTTPS** in production
2. **Set CORS origins** to specific domains
3. **Add rate limiting**
4. **Implement authentication** if needed
5. **Keep dependencies updated**
6. **Scan for vulnerabilities** regularly

## Troubleshooting

### Model not loading
- Ensure model files are in `Saved_model/` directory
- Check file permissions
- Verify file paths in environment variables

### Out of memory errors
- Reduce `max_sequences` parameter
- Increase container memory limits
- Process videos in smaller batches

### Slow predictions
- Use GPU if available (modify Dockerfile)
- Reduce video resolution before processing
- Implement caching for repeated videos
