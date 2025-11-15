"""
Fitness AI Exercise Classification System

A computer vision system for automatic exercise recognition using
MediaPipe pose estimation and deep learning.
"""

__version__ = "1.0.0"
__author__ = "Fitness AI Team"

from .models.classifier import ExerciseClassifier
from .models.pose_estimator import PoseEstimator
from .models.feature_extractor import FeatureExtractor

__all__ = [
    "ExerciseClassifier",
    "PoseEstimator",
    "FeatureExtractor",
]
