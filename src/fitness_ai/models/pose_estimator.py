"""
Advanced pose estimation using MediaPipe for biomechanical analysis
"""

import logging
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from ..config import MIN_POSE_CONFIDENCE

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Advanced pose estimation using MediaPipe for biomechanical analysis

    This class handles:
    - Real-time pose detection from video frames
    - Confidence-based filtering
    - Geometric calculations for angles and distances
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe pose estimation model

        Args:
            static_image_mode: Whether to treat each image independently
            model_complexity: Complexity of the pose model (0, 1, or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("PoseEstimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PoseEstimator: {e}")
            raise

    def extract_landmarks(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[object], float]:
        """
        Extract pose landmarks from an image frame

        Args:
            image: Input image frame from video (BGR format)

        Returns:
            tuple: (landmarks_array, pose_landmarks, average_visibility)
                - landmarks_array: Flattened array of landmark coordinates and visibility
                - pose_landmarks: MediaPipe pose landmarks object
                - average_visibility: Average visibility score across all landmarks

        Raises:
            ValueError: If image is None or invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")

        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = []
                visibility_sum = 0.0

                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])
                    visibility_sum += landmark.visibility

                # Calculate average visibility
                avg_visibility = visibility_sum / 33  # 33 landmarks

                # Filter based on average visibility
                if avg_visibility >= MIN_POSE_CONFIDENCE:
                    return np.array(landmarks), results.pose_landmarks, avg_visibility
                else:
                    logger.debug(
                        f"Pose detection confidence too low: {avg_visibility:.2f} < {MIN_POSE_CONFIDENCE}"
                    )

            return None, None, 0.0

        except cv2.error as e:
            logger.error(f"OpenCV error in extract_landmarks: {e}")
            return None, None, 0.0
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None, None, 0.0

    def calculate_angle(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> float:
        """
        Calculate angle between three points for biomechanical analysis

        Args:
            a: First point coordinates (x, y)
            b: Vertex point coordinates (x, y)
            c: Third point coordinates (x, y)

        Returns:
            float: Angle in degrees (0-360)

        Example:
            >>> estimator = PoseEstimator()
            >>> angle = estimator.calculate_angle(
            ...     np.array([1, 0]),
            ...     np.array([0, 0]),
            ...     np.array([0, 1])
            ... )
            >>> print(angle)  # Should be close to 90 degrees
        """
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            # Calculate vectors
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                     np.arctan2(a[1] - b[1], a[0] - b[0])

            angle = np.abs(radians * 180.0 / np.pi)

            # Normalize to 0-360 degrees
            return 360 - angle if angle > 180.0 else angle

        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating angle: {e}")
            return 0.0

    def calculate_distance(
        self,
        point1: np.ndarray,
        point2: np.ndarray
    ) -> float:
        """
        Calculate Euclidean distance between two points

        Args:
            point1: First point coordinates (x, y)
            point2: Second point coordinates (x, y)

        Returns:
            float: Euclidean distance between points

        Example:
            >>> estimator = PoseEstimator()
            >>> distance = estimator.calculate_distance(
            ...     np.array([0, 0]),
            ...     np.array([3, 4])
            ... )
            >>> print(distance)  # Should be 5.0
        """
        try:
            point1 = np.array(point1)
            point2 = np.array(point2)
            return float(np.sqrt(np.sum((point1 - point2) ** 2)))
        except (TypeError, ValueError) as e:
            logger.warning(f"Error calculating distance: {e}")
            return 0.0

    def draw_landmarks(
        self,
        image: np.ndarray,
        pose_landmarks: object
    ) -> np.ndarray:
        """
        Draw pose landmarks on image for visualization

        Args:
            image: Input image
            pose_landmarks: MediaPipe pose landmarks object

        Returns:
            np.ndarray: Image with drawn landmarks
        """
        try:
            if pose_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    image,
                    pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2
                    )
                )
            return image
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return image

    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
