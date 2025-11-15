"""
Utility modules for Fitness AI Exercise Classification
"""

from .video_processor import VideoProcessor, extract_features_from_video
from .logging_config import setup_logging

__all__ = [
    "VideoProcessor",
    "extract_features_from_video",
    "setup_logging",
]
