"""
Extract exercise-specific biomechanical features from pose landmarks
"""

import logging
from typing import Optional
import numpy as np

from .pose_estimator import PoseEstimator
from ..config import EXERCISE_SPECIFIC_ANGLES, FEATURE_DIM, LandmarkIndices

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract exercise-specific biomechanical features from pose landmarks

    Features include:
    - Normalized body coordinates (33 features)
    - Joint angles specific to exercise types (21 features)
    - Distance measurements (7 features)
    - Velocity tracking between frames (14 features)
    - Visibility scores (8 features)

    Total: 83 features per frame
    """

    def __init__(self):
        """Initialize feature extractor with pose estimator"""
        self.pose_estimator = PoseEstimator()
        self.prev_landmarks = None
        logger.info("FeatureExtractor initialized successfully")

    def extract_features_from_landmarks(
        self,
        landmarks: np.ndarray,
        exercise_type: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Extract comprehensive biomechanical features from pose landmarks

        Args:
            landmarks: Raw landmark data from MediaPipe (132-dimensional)
            exercise_type: Specific exercise for targeted feature extraction

        Returns:
            numpy.array: 83-dimensional feature vector or None if extraction fails

        Raises:
            ValueError: If landmarks array has invalid shape
        """
        if landmarks is None:
            logger.warning("Landmarks is None, cannot extract features")
            return None

        if len(landmarks) != 132:  # 33 landmarks * 4 values each
            raise ValueError(
                f"Invalid landmarks shape: expected 132, got {len(landmarks)}"
            )

        try:
            landmarks_reshaped = landmarks.reshape(-1, 4)
            features = []

            # Extract normalized coordinates
            coord_features = self._extract_coordinate_features(landmarks_reshaped)
            features.extend(coord_features)

            # Extract biomechanical angles
            angle_features = self._extract_angle_features(landmarks_reshaped, exercise_type)
            features.extend(angle_features)

            # Extract distance measurements
            distance_features = self._extract_distance_features(landmarks_reshaped)
            features.extend(distance_features)

            # Extract velocity features
            velocity_features = self._extract_velocity_features(landmarks_reshaped)
            features.extend(velocity_features)

            # Update previous landmarks for next frame
            self.prev_landmarks = landmarks

            # Ensure exactly 83 features
            if len(features) < FEATURE_DIM:
                features.extend([0.0] * (FEATURE_DIM - len(features)))
            elif len(features) > FEATURE_DIM:
                features = features[:FEATURE_DIM]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros(FEATURE_DIM, dtype=np.float32)

    def _extract_coordinate_features(self, landmarks_reshaped: np.ndarray) -> list:
        """
        Extract normalized coordinate features relative to hip center

        Args:
            landmarks_reshaped: Reshaped landmarks (33, 4)

        Returns:
            list: 33 normalized coordinate features (11 points * 3 coords each)
        """
        try:
            # Calculate hip center for normalization
            hip_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            ]

            # Key points for coordinate features
            key_indices = [
                LandmarkIndices.NOSE,
                LandmarkIndices.LEFT_SHOULDER,
                LandmarkIndices.RIGHT_SHOULDER,
                LandmarkIndices.LEFT_ELBOW,
                LandmarkIndices.RIGHT_ELBOW,
                LandmarkIndices.LEFT_WRIST,
                LandmarkIndices.RIGHT_WRIST,
                LandmarkIndices.LEFT_KNEE,
                LandmarkIndices.RIGHT_KNEE,
                LandmarkIndices.LEFT_ANKLE,
                LandmarkIndices.RIGHT_ANKLE
            ]

            features = []
            for idx in key_indices:
                rel_x = landmarks_reshaped[idx][0] - hip_center[0]
                rel_y = landmarks_reshaped[idx][1] - hip_center[1]
                rel_z = landmarks_reshaped[idx][2]
                features.extend([rel_x, rel_y, rel_z])

            return features

        except Exception as e:
            logger.error(f"Error extracting coordinate features: {e}")
            return [0.0] * 33

    def _extract_angle_features(
        self,
        landmarks_reshaped: np.ndarray,
        exercise_type: Optional[str] = None
    ) -> list:
        """
        Calculate biomechanical angles with exercise-specific discrimination

        Args:
            landmarks_reshaped: Reshaped landmark coordinates (33, 4)
            exercise_type: Type of exercise for specific angle calculations

        Returns:
            list: 21 biomechanical angles for exercise discrimination
        """
        try:
            angles = []

            # 1. Elbow angles (critical for all arm exercises)
            left_elbow_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_ELBOW][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2]
            )
            right_elbow_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_ELBOW][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
            )
            angles.extend([left_elbow_angle, right_elbow_angle])

            # 2. Knee angles (critical for all leg exercises)
            left_knee_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_KNEE][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][:2]
            )
            right_knee_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][:2]
            )
            angles.extend([left_knee_angle, right_knee_angle])

            # 3. Hip angles (critical for squats, deadlifts, hip thrusts)
            left_hip_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_KNEE][:2]
            )
            right_hip_angle = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][:2]
            )
            angles.extend([left_hip_angle, right_hip_angle])

            # 4. Shoulder elevation angles
            left_shoulder_elevation = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_ELBOW][:2]
            )
            right_shoulder_elevation = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_ELBOW][:2]
            )
            angles.extend([left_shoulder_elevation, right_shoulder_elevation])

            # 5. Spine/Torso angle (critical for posture)
            spine_angle = self._calculate_spine_angle(landmarks_reshaped)
            angles.append(spine_angle)

            # Exercise-specific discrimination angles
            exercise_specific = self._calculate_exercise_specific_angles(
                landmarks_reshaped, exercise_type
            )
            angles.extend(exercise_specific)

            return angles

        except Exception as e:
            logger.error(f"Error extracting angle features: {e}")
            return [0.0] * 21

    def _calculate_spine_angle(self, landmarks_reshaped: np.ndarray) -> float:
        """Calculate spine/torso angle"""
        try:
            hip_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            ]
            shoulder_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][1]) / 2
            ]
            nose = [
                landmarks_reshaped[LandmarkIndices.NOSE][0],
                landmarks_reshaped[LandmarkIndices.NOSE][1]
            ]

            return self.pose_estimator.calculate_angle(
                hip_center, shoulder_center, nose
            )
        except Exception:
            return 0.0

    def _calculate_exercise_specific_angles(
        self,
        landmarks_reshaped: np.ndarray,
        exercise_type: Optional[str]
    ) -> list:
        """
        Calculate exercise-specific angles for better discrimination

        Returns 12 exercise-specific angles
        """
        angles = []

        if not exercise_type:
            return [0.0] * 12

        try:
            specific_angles = EXERCISE_SPECIFIC_ANGLES.get(exercise_type, [])

            # Chest exercise discrimination
            if 'shoulder_horizontal' in specific_angles:
                angles.extend(self._get_chest_angles(landmarks_reshaped))
            else:
                angles.extend([0.0, 0.0])

            # Deadlift discrimination
            if exercise_type in ['deadlift', 'romanian deadlift']:
                angles.extend(self._get_deadlift_angles(landmarks_reshaped))
            else:
                angles.extend([0.0, 0.0])

            # Standing vs ground exercise discrimination
            if exercise_type in ['hammer curl', 'barbell biceps curl', 'lateral raise', 'shoulder press']:
                angles.extend(self._get_standing_angles(landmarks_reshaped))
            elif exercise_type in ['plank', 'push-up']:
                angles.extend(self._get_ground_angles(landmarks_reshaped))
            else:
                angles.extend([0.0, 0.0])

            # Curl discrimination
            if exercise_type in ['hammer curl', 'barbell biceps curl']:
                angles.extend(self._get_curl_angles(landmarks_reshaped))
            else:
                angles.extend([0.0, 0.0])

            # Additional specific angles (6 more)
            additional = self._get_additional_specific_angles(landmarks_reshaped, specific_angles)
            angles.extend(additional)

        except Exception as e:
            logger.error(f"Error calculating exercise-specific angles: {e}")
            return [0.0] * 12

        # Ensure exactly 12 angles
        if len(angles) < 12:
            angles.extend([0.0] * (12 - len(angles)))
        return angles[:12]

    def _get_chest_angles(self, landmarks_reshaped: np.ndarray) -> list:
        """Calculate chest exercise specific angles"""
        try:
            shoulder_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][1]) / 2
            ]

            arm_convergence = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                shoulder_center,
                landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
            )

            hip_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            ]

            torso_incline = self.pose_estimator.calculate_angle(
                hip_center,
                shoulder_center,
                [landmarks_reshaped[LandmarkIndices.NOSE][0],
                 landmarks_reshaped[LandmarkIndices.NOSE][1] - 0.1]
            )

            return [arm_convergence, torso_incline]
        except Exception:
            return [0.0, 0.0]

    def _get_deadlift_angles(self, landmarks_reshaped: np.ndarray) -> list:
        """Calculate deadlift specific angles"""
        try:
            left_knee = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_KNEE][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][:2]
            )
            right_knee = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][:2]
            )
            avg_knee_bend = (left_knee + right_knee) / 2

            knee_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_KNEE][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_KNEE][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][1]) / 2
            ]
            hip_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            ]

            hip_dominance = self.pose_estimator.calculate_angle(
                knee_center,
                hip_center,
                [landmarks_reshaped[LandmarkIndices.NOSE][0],
                 landmarks_reshaped[LandmarkIndices.NOSE][1]]
            )

            return [avg_knee_bend, hip_dominance]
        except Exception:
            return [0.0, 0.0]

    def _get_standing_angles(self, landmarks_reshaped: np.ndarray) -> list:
        """Calculate standing exercise specific angles"""
        try:
            shoulder_x = (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][0] +
                         landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][0]) / 2
            hip_x = (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                    landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2
            standing_posture = abs(shoulder_x - hip_x)

            hip_center = [
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                 landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            ]

            arm_position = self.pose_estimator.calculate_distance(
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                hip_center
            )

            return [standing_posture, arm_position]
        except Exception:
            return [0.0, 0.0]

    def _get_ground_angles(self, landmarks_reshaped: np.ndarray) -> list:
        """Calculate ground exercise specific angles"""
        try:
            nose_y = landmarks_reshaped[LandmarkIndices.NOSE][1]
            hip_y = (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                    landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
            body_parallel = abs(nose_y - hip_y)

            support_width = self.pose_estimator.calculate_distance(
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
            )

            return [body_parallel, support_width]
        except Exception:
            return [0.0, 0.0]

    def _get_curl_angles(self, landmarks_reshaped: np.ndarray) -> list:
        """Calculate curl exercise specific angles"""
        try:
            wrist_alignment = self.pose_estimator.calculate_angle(
                landmarks_reshaped[LandmarkIndices.LEFT_ELBOW][:2],
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                [landmarks_reshaped[LandmarkIndices.LEFT_WRIST][0] + 0.1,
                 landmarks_reshaped[LandmarkIndices.LEFT_WRIST][1]]
            )

            hand_separation = self.pose_estimator.calculate_distance(
                landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
            )

            return [wrist_alignment, hand_separation]
        except Exception:
            return [0.0, 0.0]

    def _get_additional_specific_angles(
        self,
        landmarks_reshaped: np.ndarray,
        specific_angles: list
    ) -> list:
        """Calculate additional exercise-specific angles (6 features)"""
        angles = []

        try:
            # Shoulder horizontal
            if 'shoulder_horizontal' in specific_angles:
                angle = self.pose_estimator.calculate_angle(
                    landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2]
                )
                angles.append(angle)
            else:
                angles.append(0.0)

            # Ankle angles
            if 'ankle_angles' in specific_angles:
                left_ankle = self.pose_estimator.calculate_angle(
                    landmarks_reshaped[LandmarkIndices.LEFT_KNEE][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][:2],
                    [landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][0] + 0.1,
                     landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][1]]
                )
                right_ankle = self.pose_estimator.calculate_angle(
                    landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][:2],
                    [landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][0] + 0.1,
                     landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][1]]
                )
                angles.extend([left_ankle, right_ankle])
            else:
                angles.extend([0.0, 0.0])

            # Hip hinge
            if 'hip_hinge' in specific_angles:
                knee_center = [
                    (landmarks_reshaped[LandmarkIndices.LEFT_KNEE][0] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][0]) / 2,
                    (landmarks_reshaped[LandmarkIndices.LEFT_KNEE][1] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_KNEE][1]) / 2
                ]
                hip_center = [
                    (landmarks_reshaped[LandmarkIndices.LEFT_HIP][0] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_HIP][0]) / 2,
                    (landmarks_reshaped[LandmarkIndices.LEFT_HIP][1] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_HIP][1]) / 2
                ]
                shoulder_center = [
                    (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][0] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][0]) / 2,
                    (landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][1] +
                     landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][1]) / 2
                ]

                hip_hinge = self.pose_estimator.calculate_angle(
                    knee_center, hip_center, shoulder_center
                )
                angles.append(hip_hinge)
            else:
                angles.append(0.0)

            # Shoulder abduction
            if 'shoulder_abduction' in specific_angles:
                left_abduction = self.pose_estimator.calculate_angle(
                    landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2]
                )
                right_abduction = self.pose_estimator.calculate_angle(
                    landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
                )
                angles.extend([left_abduction, right_abduction])
            else:
                angles.extend([0.0, 0.0])

        except Exception as e:
            logger.error(f"Error in additional specific angles: {e}")
            return [0.0] * 6

        # Ensure exactly 6 angles
        if len(angles) < 6:
            angles.extend([0.0] * (6 - len(angles)))
        return angles[:6]

    def _extract_distance_features(self, landmarks_reshaped: np.ndarray) -> list:
        """
        Extract biomechanical distance measurements

        Returns:
            list: 7 distance features
        """
        try:
            distances = [
                # Wrist distance
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
                ),
                # Ankle distance
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][:2]
                ),
                # Shoulder distance
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2]
                ),
                # Left arm reach
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.LEFT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_WRIST][:2]
                ),
                # Right arm reach
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.RIGHT_SHOULDER][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_WRIST][:2]
                ),
                # Left leg length
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.LEFT_HIP][:2],
                    landmarks_reshaped[LandmarkIndices.LEFT_ANKLE][:2]
                ),
                # Right leg length
                self.pose_estimator.calculate_distance(
                    landmarks_reshaped[LandmarkIndices.RIGHT_HIP][:2],
                    landmarks_reshaped[LandmarkIndices.RIGHT_ANKLE][:2]
                )
            ]
            return distances
        except Exception as e:
            logger.error(f"Error extracting distance features: {e}")
            return [0.0] * 7

    def _extract_velocity_features(self, landmarks_reshaped: np.ndarray) -> list:
        """
        Extract velocity features for movement dynamics

        Returns:
            list: 14 velocity features (7 points * 2 coordinates)
        """
        if self.prev_landmarks is None:
            return [0.0] * 14

        try:
            prev_reshaped = self.prev_landmarks.reshape(-1, 4)
            velocities = []

            # Upper body + core movement indices
            key_indices = [
                LandmarkIndices.NOSE,
                LandmarkIndices.LEFT_SHOULDER,
                LandmarkIndices.RIGHT_SHOULDER,
                LandmarkIndices.LEFT_ELBOW,
                LandmarkIndices.RIGHT_ELBOW,
                LandmarkIndices.LEFT_WRIST,
                LandmarkIndices.RIGHT_WRIST
            ]

            for idx in key_indices:
                vel_x = landmarks_reshaped[idx][0] - prev_reshaped[idx][0]
                vel_y = landmarks_reshaped[idx][1] - prev_reshaped[idx][1]
                velocities.extend([vel_x, vel_y])

            return velocities

        except Exception as e:
            logger.error(f"Error extracting velocity features: {e}")
            return [0.0] * 14

    def reset(self):
        """Reset previous landmarks (useful for processing new video)"""
        self.prev_landmarks = None
        logger.debug("Feature extractor reset")
