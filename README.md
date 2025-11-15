# Fitness AI Exercise Classification System

<img width="215" height="241" alt="train2" src="https://github.com/user-attachments/assets/f4915d48-fd0b-4fe7-9a96-2fbb9498fd27" />

[![CI/CD Pipeline](https://github.com/HayLahav/Fitness-AI-Exercise-Classification-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/HayLahav/Fitness-AI-Exercise-Classification-Project/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📌 Overview
This project presents an AI-powered system that automatically recognizes **22 different exercise types** from workout videos.
It combines **pose estimation**, **biomechanical feature extraction**, and **sequence modeling** to deliver robust, real-world exercise classification — even in challenging conditions such as varying lighting, camera angles, and backgrounds.

The model is designed for:
- **Personal fitness tracking**
- **Virtual coaching & online training platforms**
- **Rehabilitation and sports performance analysis**

### ✨ New: Production-Ready Features
- 🏗️ **Modular Architecture**: Refactored from notebook to production-ready Python package
- 🚀 **REST API**: FastAPI-based API for easy integration
- 🐳 **Docker Support**: Containerized deployment with Docker Compose
- ✅ **Testing**: Comprehensive test suite with pytest
- 📝 **Logging**: Professional logging framework throughout
- 🔒 **Error Handling**: Robust error handling and input validation
- 🔄 **CI/CD**: Automated testing and deployment with GitHub Actions

---

## 📊 Dataset
- **Source:** Public workout videos from YouTube
- **Categories:** 14 upper body, 6 lower body, and 2 core exercises
- **Diversity:** Multiple environments, camera angles, and participant demographics
- **Challenges:** Class imbalance, partial body coverage, motion blur, and background distractions

---

## 📈 Key Results
- **Test Accuracy:** 79.1%
- **Balanced Accuracy:** 72.8%
- **Cohen's Kappa:** 0.792 (substantial agreement)
- **High performers:** Push-Up (92%), Tricep Pushdown (94%), Lateral Raise (88%)
- **Challenging cases:** Romanian Deadlift, Hammer Curl, Pull-Up

The system demonstrates strong performance for most exercises and provides realistic generalization to unseen videos.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HayLahav/Fitness-AI-Exercise-Classification-Project.git
cd Fitness-AI-Exercise-Classification-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Using the REST API

#### Start the API server:

```bash
# Development mode
uvicorn fitness_ai.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn fitness_ai.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Or using Docker:

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### API Usage Examples:

```bash
# Health check
curl http://localhost:8000/health

# List supported exercises
curl http://localhost:8000/exercises

# Classify a video
curl -X POST "http://localhost:8000/classify" \
  -F "file=@path/to/your/video.mp4" \
  -F "max_sequences=10"
```

#### API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Using the Python Package

```python
from fitness_ai.models import ExerciseClassifier

# Initialize classifier
classifier = ExerciseClassifier(
    model_path="Saved_model/exercise_classifier_model.keras",
    label_encoder_path="Saved_model/label_encoder.pkl",
    scaler_path="Saved_model/scaler.pkl"
)

# Classify a video
result = classifier.classify_video("squat_video.mp4")

if result:
    exercise, confidence, all_preds, all_confs = result
    print(f"Exercise: {exercise}")
    print(f"Confidence: {confidence:.2%}")
```

---

## 🏗️ Project Structure

```
Fitness-AI-Exercise-Classification-Project/
├── src/fitness_ai/              # Main package
│   ├── models/                  # ML models
│   │   ├── pose_estimator.py   # MediaPipe pose estimation
│   │   ├── feature_extractor.py # Feature extraction
│   │   ├── classifier.py        # Exercise classifier
│   │   └── model_architecture.py # Model architecture
│   ├── data/                    # Data processing
│   │   └── augmentation.py      # Data augmentation
│   ├── training/                # Training utilities
│   │   └── loss_functions.py    # Custom loss functions
│   ├── utils/                   # Utilities
│   │   ├── video_processor.py   # Video processing
│   │   └── logging_config.py    # Logging setup
│   ├── api/                     # REST API
│   │   └── main.py              # FastAPI application
│   └── config.py                # Configuration
├── tests/                       # Test suite
├── notebooks/                   # Original research notebook
├── Saved_model/                 # Trained model files
├── .github/workflows/           # CI/CD pipelines
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Docker Compose config
└── README.md                    # This file
```

---

## 🔧 How It Works

### 1. Pose Estimation
Uses **MediaPipe** to detect 33 body landmarks from video frames with confidence-based filtering.

### 2. Feature Extraction (83 features)
- **Normalized coordinates** (33 features): Body landmarks relative to hip center
- **Joint angles** (21 features): Elbow, knee, hip, shoulder, spine angles
- **Distance measurements** (7 features): Body segment distances
- **Velocity features** (14 features): Movement dynamics between frames
- **Visibility scores** (8 features): Landmark detection confidence

### 3. Sequence Analysis
**Bidirectional LSTM with Multi-Head Attention**:
- Temporal pattern recognition across 20-frame sequences
- Attention mechanism focuses on discriminative movements
- Global pooling aggregates sequence-level features

### 4. Classification
- **Ensemble voting** across multiple sequences
- **Confidence scoring** for prediction reliability
- **Focal loss** handles class imbalance

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/fitness_ai --cov-report=html

# Run specific test file
pytest tests/test_pose_estimator.py -v
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t fitness-ai:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/Saved_model:/app/Saved_model:ro \
  fitness-ai:latest

# Using Docker Compose (recommended)
docker-compose up -d
```

---

## 📚 Supported Exercises

**Upper Body (14):**
- Barbell Biceps Curl
- Hammer Curl
- Tricep Pushdown
- Tricep Dips
- Bench Press
- Incline Bench Press
- Decline Bench Press
- Shoulder Press
- Lat Pulldown
- Pull Up
- T Bar Row
- Lateral Raise
- Chest Fly Machine
- Push-Up

**Lower Body (6):**
- Squat
- Deadlift
- Romanian Deadlift
- Leg Extension
- Hip Thrust

**Core (2):**
- Plank
- Russian Twist
- Leg Raises

---

## 💡 Key Contributions
- Handles **real-world video variability** without requiring controlled environments
- Addresses **class imbalance** using focal loss and class weighting
- Uses **confidence-based filtering** to remove low-quality pose detections
- Employs **ensemble predictions** for robust final classification
- **Production-ready architecture** with API, testing, and containerization

---

## 🔮 Future Directions
- Add **vision-language models** for context-aware recognition
- Combine **pose data with visual cues** for better equipment-based classification
- Deploy on **mobile and edge devices** for real-time exercise tracking
- Extend to **form assessment and injury prevention feedback**
- Implement **real-time webcam inference**
- Add **model monitoring and drift detection**

---

## 📄 License

MIT License - see LICENSE file for details

---

## 📬 Contact
**Author:** Hay Lahav
📧 Email: haylahav1@gmail.com

---

## 🙏 Acknowledgments

- MediaPipe for pose estimation
- TensorFlow/Keras for deep learning framework
- FastAPI for API framework
