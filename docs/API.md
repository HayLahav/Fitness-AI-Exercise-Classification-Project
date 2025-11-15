# API Documentation

## Overview

The Fitness AI Exercise Classification API provides REST endpoints for classifying exercise types from video files.

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET /exercises

List all supported exercise classes.

**Response:**
```json
{
  "exercises": [
    "barbell biceps curl",
    "bench press",
    ...
  ],
  "count": 22
}
```

### POST /classify

Classify exercise from video file.

**Parameters:**
- `file` (form-data): Video file (mp4, avi, mov, mkv)
- `max_sequences` (query, optional): Maximum sequences to analyze (default: 10, max: 50)

**Request Example:**
```bash
curl -X POST "http://localhost:8000/classify?max_sequences=10" \
  -F "file=@squat_video.mp4"
```

**Response:**
```json
{
  "exercise": "squat",
  "confidence": 0.89,
  "all_predictions": ["squat", "squat", "deadlift"],
  "all_confidences": [0.92, 0.88, 0.78],
  "num_sequences": 3
}
```

**Error Responses:**

- 400: Invalid file type
- 422: No valid pose sequences detected
- 503: Model not loaded

## Python Client Example

```python
import requests

# Classify video
with open("squat_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/classify",
        files={"file": f},
        params={"max_sequences": 10}
    )

if response.status_code == 200:
    result = response.json()
    print(f"Exercise: {result['exercise']}")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
