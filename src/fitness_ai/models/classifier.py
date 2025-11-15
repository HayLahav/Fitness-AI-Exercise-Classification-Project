"""
Complete exercise classification system for inference
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, List
from collections import defaultdict
import numpy as np
import tensorflow as tf

from .feature_extractor import FeatureExtractor
from ..utils.video_processor import extract_features_from_video
from ..training.loss_functions import focal_loss
from ..config import EXERCISE_CLASSES, MODEL_SAVE_PATH

logger = logging.getLogger(__name__)


class ExerciseClassifier:
    """
    Complete exercise classification system for inference

    This class provides:
    - Video loading and processing
    - Pose estimation and feature extraction
    - Exercise prediction with confidence scores
    - Ensemble voting for robust predictions
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        """
        Initialize classifier with trained model components

        Args:
            model_path: Path to saved Keras model (.keras or .h5)
            label_encoder_path: Path to saved label encoder (.pkl)
            scaler_path: Path to saved feature scaler (.pkl)

        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If model files are invalid
        """
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.model_loaded = False

        # Default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_PATH, 'exercise_classifier_model.keras')
        if label_encoder_path is None:
            label_encoder_path = os.path.join(MODEL_SAVE_PATH, 'label_encoder.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')

        # Load model components
        if model_path and os.path.exists(model_path):
            try:
                self._load_model_components(model_path, label_encoder_path, scaler_path)
                logger.info("Exercise classifier loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Classifier initialized without model (training mode)")

    def _load_model_components(
        self,
        model_path: str,
        label_encoder_path: str,
        scaler_path: str
    ):
        """Load model, label encoder, and scaler"""
        # Load model with custom objects
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fn': focal_loss()}
        )

        # Load label encoder
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

        logger.info(f"Loading label encoder from {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        logger.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.model_loaded = True

    def classify_video(
        self,
        video_path: str,
        max_sequences: int = 10
    ) -> Optional[Tuple[str, float, List[str], List[float]]]:
        """
        Classify exercise type from video file

        Args:
            video_path: Path to video file
            max_sequences: Maximum sequences to analyze

        Returns:
            tuple: (prediction, confidence, all_predictions, all_confidences)
                - prediction: Predicted exercise name
                - confidence: Average confidence score
                - all_predictions: List of predictions from each sequence
                - all_confidences: List of confidence scores
            Returns None if classification fails

        Example:
            >>> classifier = ExerciseClassifier()
            >>> result = classifier.classify_video("squat.mp4")
            >>> if result:
            ...     exercise, conf, _, _ = result
            ...     print(f"Exercise: {exercise}, Confidence: {conf:.2%}")
        """
        if not self.model_loaded:
            logger.error("Model not loaded. Cannot classify video.")
            return None

        # Try to detect exercise type from filename for better feature extraction
        detected_exercise = self._detect_exercise_from_filename(video_path)

        # Extract features from video
        try:
            logger.info(f"Extracting features from {video_path}")
            sequences = extract_features_from_video(
                video_path,
                self.feature_extractor,
                max_sequences=max_sequences,
                exercise_type=detected_exercise
            )
        except Exception as e:
            logger.error(f"Error extracting features from video: {e}")
            return None

        if not sequences:
            logger.warning(f"No valid sequences extracted from {video_path}")
            return None

        logger.info(f"Extracted {len(sequences)} sequences, performing classification")

        predictions = []
        confidences = []

        # Process each sequence
        for i, sequence in enumerate(sequences):
            try:
                # Reshape for model input: (1, timesteps, features)
                sequence_reshaped = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

                # Scale features
                sequence_scaled = self.scaler.transform(
                    sequence_reshaped.reshape(-1, sequence_reshaped.shape[-1])
                ).reshape(sequence_reshaped.shape)

                # Predict
                prediction = self.model.predict(sequence_scaled, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                # Convert to exercise name
                exercise_name = self.label_encoder.inverse_transform([predicted_class])[0]
                predictions.append(exercise_name)
                confidences.append(float(confidence))

                logger.debug(
                    f"Sequence {i+1}/{len(sequences)}: {exercise_name} ({confidence:.2%})"
                )

            except Exception as e:
                logger.error(f"Error predicting sequence {i+1}: {e}")
                continue

        if not predictions:
            logger.error("No successful predictions")
            return None

        # Weighted voting based on confidence
        final_prediction, avg_confidence = self._ensemble_predictions(
            predictions, confidences
        )

        logger.info(
            f"Final prediction: {final_prediction} "
            f"(confidence: {avg_confidence:.2%})"
        )

        return final_prediction, avg_confidence, predictions, confidences

    def _detect_exercise_from_filename(self, video_path: str) -> Optional[str]:
        """
        Try to detect exercise type from video filename

        Args:
            video_path: Path to video file

        Returns:
            Detected exercise name or None
        """
        video_name = Path(video_path).stem.lower()

        for exercise in EXERCISE_CLASSES:
            exercise_key = exercise.lower().replace(' ', '_')
            if exercise_key in video_name or exercise.lower() in video_name:
                logger.debug(f"Detected exercise from filename: {exercise}")
                return exercise

        return None

    def _ensemble_predictions(
        self,
        predictions: List[str],
        confidences: List[float]
    ) -> Tuple[str, float]:
        """
        Combine multiple predictions using weighted voting

        Args:
            predictions: List of predicted exercise names
            confidences: List of confidence scores

        Returns:
            tuple: (final_prediction, average_confidence)
        """
        # Weighted voting based on confidence
        prediction_weights = defaultdict(float)
        for pred, conf in zip(predictions, confidences):
            prediction_weights[pred] += conf

        # Get prediction with highest total weight
        final_prediction = max(prediction_weights, key=prediction_weights.get)

        # Calculate average confidence for final prediction
        final_confidences = [
            conf for pred, conf in zip(predictions, confidences)
            if pred == final_prediction
        ]
        avg_confidence = float(np.mean(final_confidences))

        return final_prediction, avg_confidence

    def predict_features(
        self,
        features: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict exercise from pre-extracted features

        Args:
            features: Feature array of shape (timesteps, feature_dim) or
                     (batch, timesteps, feature_dim)

        Returns:
            tuple: (exercise_name, confidence, probabilities)

        Raises:
            ValueError: If model not loaded or invalid feature shape
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")

        # Handle both single sequence and batch
        if len(features.shape) == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])

        # Scale features
        original_shape = features.shape
        features_scaled = self.scaler.transform(
            features.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        # Predict
        probabilities = self.model.predict(features_scaled, verbose=0)

        # Get prediction
        predicted_class = np.argmax(probabilities[0])
        confidence = float(np.max(probabilities[0]))
        exercise_name = self.label_encoder.inverse_transform([predicted_class])[0]

        return exercise_name, confidence, probabilities[0]

    def get_supported_exercises(self) -> List[str]:
        """
        Get list of supported exercise classes

        Returns:
            List of exercise names
        """
        if self.label_encoder:
            return list(self.label_encoder.classes_)
        return EXERCISE_CLASSES
