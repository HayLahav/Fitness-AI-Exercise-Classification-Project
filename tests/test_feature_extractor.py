"""
Unit tests for FeatureExtractor
"""

import pytest
import numpy as np

from fitness_ai.models.feature_extractor import FeatureExtractor
from fitness_ai.config import FEATURE_DIM


@pytest.fixture
def feature_extractor():
    """Create FeatureExtractor instance for testing"""
    return FeatureExtractor()


@pytest.fixture
def sample_landmarks():
    """Create sample landmarks (33 landmarks * 4 values = 132)"""
    return np.random.rand(132)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class"""

    def test_initialization(self, feature_extractor):
        """Test that FeatureExtractor initializes correctly"""
        assert feature_extractor is not None
        assert feature_extractor.pose_estimator is not None
        assert feature_extractor.prev_landmarks is None

    def test_extract_features_valid_landmarks(self, feature_extractor, sample_landmarks):
        """Test feature extraction with valid landmarks"""
        features = feature_extractor.extract_features_from_landmarks(sample_landmarks)

        assert features is not None
        assert len(features) == FEATURE_DIM
        assert features.dtype == np.float32

    def test_extract_features_invalid_shape(self, feature_extractor):
        """Test feature extraction with invalid landmark shape"""
        invalid_landmarks = np.random.rand(100)  # Wrong size

        with pytest.raises(ValueError):
            feature_extractor.extract_features_from_landmarks(invalid_landmarks)

    def test_extract_features_none_landmarks(self, feature_extractor):
        """Test feature extraction with None landmarks"""
        features = feature_extractor.extract_features_from_landmarks(None)
        assert features is None

    def test_extract_features_with_exercise_type(self, feature_extractor, sample_landmarks):
        """Test feature extraction with specific exercise type"""
        features = feature_extractor.extract_features_from_landmarks(
            sample_landmarks,
            exercise_type="squat"
        )

        assert features is not None
        assert len(features) == FEATURE_DIM

    def test_feature_consistency(self, feature_extractor, sample_landmarks):
        """Test that same landmarks produce same features"""
        features1 = feature_extractor.extract_features_from_landmarks(sample_landmarks)

        # Reset to clear prev_landmarks
        feature_extractor.reset()

        features2 = feature_extractor.extract_features_from_landmarks(sample_landmarks)

        np.testing.assert_array_almost_equal(features1, features2)

    def test_reset(self, feature_extractor, sample_landmarks):
        """Test reset functionality"""
        feature_extractor.extract_features_from_landmarks(sample_landmarks)
        assert feature_extractor.prev_landmarks is not None

        feature_extractor.reset()
        assert feature_extractor.prev_landmarks is None
