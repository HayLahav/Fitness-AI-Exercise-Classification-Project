"""
Video processing utilities for exercise classification
"""

import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from ..config import SEQUENCE_LENGTH, FRAME_SKIP, MAX_SEQUENCES_PER_VIDEO
from ..models.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Process video files to extract exercise features

    Handles:
    - Video file loading and validation
    - Frame extraction with sampling
    - Feature extraction from poses
    - Sequence creation for model input
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        frame_skip: int = FRAME_SKIP
    ):
        """
        Initialize video processor

        Args:
            feature_extractor: Feature extractor instance
            sequence_length: Number of frames per sequence
            frame_skip: Number of frames to skip between samples
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        logger.info(
            f"VideoProcessor initialized: sequence_length={sequence_length}, "
            f"frame_skip={frame_skip}"
        )

    def process_video(
        self,
        video_path: str,
        exercise_type: Optional[str] = None,
        max_sequences: int = MAX_SEQUENCES_PER_VIDEO
    ) -> List[np.ndarray]:
        """
        Process video file and extract feature sequences

        Args:
            video_path: Path to video file
            exercise_type: Type of exercise for targeted feature extraction
            max_sequences: Maximum number of sequences to extract

        Returns:
            List of feature sequences, each of shape (sequence_length, feature_dim)

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        # Validate video file
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            logger.warning(f"Unusual video format: {video_file.suffix}")

        logger.info(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            sequences = self._extract_sequences(
                cap, exercise_type, max_sequences
            )
            logger.info(f"Extracted {len(sequences)} sequences from {video_path}")
            return sequences

        finally:
            cap.release()

    def _extract_sequences(
        self,
        cap: cv2.VideoCapture,
        exercise_type: Optional[str],
        max_sequences: int
    ) -> List[np.ndarray]:
        """
        Extract feature sequences from video capture

        Args:
            cap: OpenCV video capture object
            exercise_type: Type of exercise
            max_sequences: Maximum sequences to extract

        Returns:
            List of feature sequences
        """
        features_list = []
        sequences = []
        frame_count = 0

        # Reset feature extractor for new video
        self.feature_extractor.reset()

        while cap.isOpened() and len(sequences) < max_sequences:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip frames for efficiency
            if frame_count % self.frame_skip != 0:
                continue

            # Extract pose and features
            try:
                landmarks, _, visibility = self.feature_extractor.pose_estimator.extract_landmarks(frame)

                if landmarks is not None:
                    features = self.feature_extractor.extract_features_from_landmarks(
                        landmarks, exercise_type
                    )

                    if features is not None:
                        features_list.append(features)

                        # Create sequence when we have enough frames
                        if len(features_list) >= self.sequence_length:
                            sequence = np.array(features_list[-self.sequence_length:])
                            sequences.append(sequence)
                            logger.debug(
                                f"Created sequence {len(sequences)} "
                                f"at frame {frame_count}"
                            )

            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {e}")
                continue

        if not sequences and features_list:
            # If we have some features but not enough for a full sequence, pad
            logger.warning(
                f"Only {len(features_list)} frames extracted, "
                f"need {self.sequence_length}. Padding sequence."
            )
            if len(features_list) > 0:
                sequence = self._pad_sequence(features_list)
                if sequence is not None:
                    sequences.append(sequence)

        return sequences

    def _pad_sequence(self, features_list: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Pad feature list to create a valid sequence

        Args:
            features_list: List of feature arrays

        Returns:
            Padded sequence or None if list is empty
        """
        if not features_list:
            return None

        # Repeat the sequence to reach required length
        if len(features_list) < self.sequence_length:
            repeats = (self.sequence_length // len(features_list)) + 1
            padded = (features_list * repeats)[:self.sequence_length]
            return np.array(padded)

        return np.array(features_list[:self.sequence_length])


def extract_features_from_video(
    video_path: str,
    feature_extractor: FeatureExtractor,
    max_sequences: int = MAX_SEQUENCES_PER_VIDEO,
    exercise_type: Optional[str] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    frame_skip: int = FRAME_SKIP
) -> List[np.ndarray]:
    """
    Convenience function to extract features from a video

    Args:
        video_path: Path to video file
        feature_extractor: Feature extractor instance
        max_sequences: Maximum sequences to extract
        exercise_type: Type of exercise
        sequence_length: Frames per sequence
        frame_skip: Frame sampling rate

    Returns:
        List of feature sequences

    Example:
        >>> from fitness_ai.models import FeatureExtractor
        >>> extractor = FeatureExtractor()
        >>> sequences = extract_features_from_video(
        ...     "squat_video.mp4",
        ...     extractor,
        ...     exercise_type="squat"
        ... )
    """
    processor = VideoProcessor(
        feature_extractor=feature_extractor,
        sequence_length=sequence_length,
        frame_skip=frame_skip
    )

    return processor.process_video(
        video_path=video_path,
        exercise_type=exercise_type,
        max_sequences=max_sequences
    )
