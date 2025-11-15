"""
Configuration and constants for Exercise Classification System
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset configuration
DATASET_PATH = os.getenv('DATASET_PATH', str(PROJECT_ROOT / 'data' / 'WorkoutFitness'))
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', str(PROJECT_ROOT / 'Saved_model'))

# Video processing configuration
SEQUENCE_LENGTH = 20  # Number of frames per sequence
FRAME_SKIP = 2        # Frame sampling rate for efficiency
MIN_POSE_CONFIDENCE = 0.6  # Minimum confidence for pose detection
MAX_SEQUENCES_PER_VIDEO = 10  # Maximum sequences to extract from a single video

# Feature extraction configuration
FEATURE_DIM = 83  # Dimension of extracted feature vectors
NUM_LANDMARKS = 33  # Number of MediaPipe pose landmarks

# Model configuration
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 32
DENSE_UNITS_1 = 64
DENSE_UNITS_2 = 32
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2

# Training configuration
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Focal loss parameters
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0

# Complete list of supported exercise classes
EXERCISE_CLASSES = [
    'barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift',
    'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press',
    'lat pulldown', 'lateral raise', 'leg extension', 'leg raises',
    'plank', 'pull Up', 'push-up', 'romanian deadlift',
    'russian twist', 'shoulder press', 'squat', 't bar row',
    'tricep Pushdown', 'tricep dips'
]

# Exercise-specific biomechanical angle definitions for intelligent discrimination
EXERCISE_SPECIFIC_ANGLES = {
    # Upper body exercises - focus on arm and shoulder angles
    'barbell biceps curl': ['elbow_angles', 'shoulder_elevation', 'wrist_stability'],
    'hammer curl': ['elbow_angles', 'forearm_rotation', 'shoulder_stability'],
    'tricep Pushdown': ['elbow_angles', 'shoulder_extension', 'torso_stability'],
    'tricep dips': ['elbow_angles', 'shoulder_depression', 'hip_angle'],

    # Pressing movements - focus on arm extension and shoulder mechanics
    'bench press': ['elbow_angles', 'shoulder_horizontal', 'chest_expansion'],
    'incline bench press': ['elbow_angles', 'shoulder_incline', 'upper_chest_angle'],
    'decline bench press': ['elbow_angles', 'shoulder_decline', 'lower_chest_angle'],
    'shoulder press': ['elbow_angles', 'shoulder_elevation', 'core_stability'],

    # Pulling movements - focus on lat and rhomboid engagement
    'lat pulldown': ['elbow_angles', 'shoulder_adduction', 'lat_stretch'],
    'pull Up': ['elbow_angles', 'shoulder_adduction', 'hanging_posture'],
    't bar row': ['elbow_angles', 'shoulder_retraction', 'hip_hinge'],

    # Leg exercises - focus on hip, knee, and ankle mechanics
    'squat': ['knee_angles', 'hip_angles', 'ankle_angles', 'spine_angle'],
    'deadlift': ['knee_angles', 'hip_angles', 'spine_angle', 'shoulder_position'],
    'romanian deadlift': ['hip_angles', 'knee_slight_bend', 'spine_neutral', 'hamstring_stretch'],
    'leg extension': ['knee_angles', 'hip_stability', 'quad_isolation'],
    'hip thrust': ['hip_angles', 'knee_angles', 'glute_activation'],

    # Core and stability exercises
    'plank': ['spine_alignment', 'hip_stability', 'shoulder_stability'],
    'leg raises': ['hip_flexion', 'knee_angles', 'core_engagement'],
    'russian twist': ['spine_rotation', 'hip_stability', 'core_twist'],

    # Machine exercises
    'chest fly machine': ['shoulder_horizontal', 'elbow_slight_bend', 'chest_stretch'],
    'lateral raise': ['shoulder_abduction', 'elbow_slight_bend', 'deltoid_isolation'],

    # Bodyweight exercises
    'push-up': ['elbow_angles', 'shoulder_stability', 'plank_position', 'hip_alignment']
}

# MediaPipe landmark indices
class LandmarkIndices:
    """MediaPipe pose landmark indices for easy reference"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
