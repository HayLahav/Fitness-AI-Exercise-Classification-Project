"""
FastAPI REST API for Exercise Classification System
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, List
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..models.classifier import ExerciseClassifier
from ..utils.logging_config import setup_logging
from ..config import EXERCISE_CLASSES

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Fitness AI Exercise Classification API",
    description="AI-powered exercise classification from video using pose estimation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier: Optional[ExerciseClassifier] = None


# Pydantic models for API
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


class ExerciseListResponse(BaseModel):
    """List of supported exercises"""
    exercises: List[str]
    count: int


class ClassificationResult(BaseModel):
    """Exercise classification result"""
    exercise: str = Field(..., description="Predicted exercise name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    all_predictions: List[str] = Field(..., description="Predictions from all sequences")
    all_confidences: List[float] = Field(..., description="Confidence scores for all sequences")
    num_sequences: int = Field(..., description="Number of sequences analyzed")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize classifier on startup"""
    global classifier
    try:
        logger.info("Initializing Exercise Classifier...")
        classifier = ExerciseClassifier()
        logger.info("Classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        logger.warning("API starting without loaded model")
        # Don't fail startup, but classifier will be None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint - returns API health status
    """
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.model_loaded,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Health status and model loading status
    """
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.model_loaded,
        "version": "1.0.0"
    }


@app.get("/exercises", response_model=ExerciseListResponse)
async def list_exercises():
    """
    Get list of supported exercise classes

    Returns:
        List of exercise names that can be classified
    """
    if classifier:
        exercises = classifier.get_supported_exercises()
    else:
        exercises = EXERCISE_CLASSES

    return {
        "exercises": exercises,
        "count": len(exercises)
    }


@app.post("/classify", response_model=ClassificationResult)
async def classify_video(
    file: UploadFile = File(..., description="Video file to classify"),
    max_sequences: int = Query(10, ge=1, le=50, description="Maximum sequences to analyze")
):
    """
    Classify exercise type from uploaded video

    Args:
        file: Video file (mp4, avi, mov, mkv)
        max_sequences: Maximum number of sequences to extract and analyze

    Returns:
        Classification result with exercise name and confidence

    Raises:
        HTTPException: If classification fails or model not loaded
    """
    if classifier is None or not classifier.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are available."
        )

    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as temp_file:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        logger.info(f"Processing video: {file.filename}")

        # Classify video
        result = classifier.classify_video(temp_path, max_sequences=max_sequences)

        if result is None:
            raise HTTPException(
                status_code=422,
                detail="Failed to classify video. No valid pose sequences detected."
            )

        exercise, confidence, all_preds, all_confs = result

        return ClassificationResult(
            exercise=exercise,
            confidence=confidence,
            all_predictions=all_preds,
            all_confidences=all_confs,
            num_sequences=len(all_preds)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying video: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
