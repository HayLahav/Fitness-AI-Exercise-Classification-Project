"""
Models module for Fitness AI Exercise Classification
"""

from .pose_estimator import PoseEstimator
from .feature_extractor import FeatureExtractor
from .classifier import ExerciseClassifier
from .model_architecture import create_attention_model

__all__ = [
    "PoseEstimator",
    "FeatureExtractor",
    "ExerciseClassifier",
    "create_attention_model",
]
