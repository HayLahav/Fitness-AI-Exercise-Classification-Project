"""
Unit tests for PoseEstimator
"""

import pytest
import numpy as np
import cv2

from fitness_ai.models.pose_estimator import PoseEstimator


@pytest.fixture
def pose_estimator():
    """Create PoseEstimator instance for testing"""
    return PoseEstimator()


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a blank image (black)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return image


class TestPoseEstimator:
    """Test cases for PoseEstimator class"""

    def test_initialization(self, pose_estimator):
        """Test that PoseEstimator initializes correctly"""
        assert pose_estimator is not None
        assert pose_estimator.pose is not None
        assert pose_estimator.mp_pose is not None

    def test_extract_landmarks_with_valid_image(self, pose_estimator, sample_image):
        """Test landmark extraction with a valid image"""
        landmarks, pose_landmarks, visibility = pose_estimator.extract_landmarks(sample_image)

        # For blank image, we likely won't detect a pose, so landmarks could be None
        # This test just ensures the method runs without error
        assert isinstance(visibility, float)
        assert 0.0 <= visibility <= 1.0

    def test_extract_landmarks_with_invalid_image(self, pose_estimator):
        """Test landmark extraction with invalid image"""
        with pytest.raises(ValueError):
            pose_estimator.extract_landmarks(None)

    def test_calculate_angle(self, pose_estimator):
        """Test angle calculation between three points"""
        # Right angle test
        a = np.array([1, 0])
        b = np.array([0, 0])
        c = np.array([0, 1])

        angle = pose_estimator.calculate_angle(a, b, c)
        assert 85 <= angle <= 95  # Should be close to 90 degrees

    def test_calculate_angle_straight_line(self, pose_estimator):
        """Test angle calculation for straight line"""
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([2, 0])

        angle = pose_estimator.calculate_angle(a, b, c)
        assert angle <= 5 or angle >= 175  # Should be close to 0 or 180

    def test_calculate_distance(self, pose_estimator):
        """Test distance calculation between two points"""
        point1 = np.array([0, 0])
        point2 = np.array([3, 4])

        distance = pose_estimator.calculate_distance(point1, point2)
        assert abs(distance - 5.0) < 0.01  # 3-4-5 triangle

    def test_calculate_distance_same_point(self, pose_estimator):
        """Test distance calculation for same point"""
        point = np.array([1, 1])

        distance = pose_estimator.calculate_distance(point, point)
        assert distance == 0.0
