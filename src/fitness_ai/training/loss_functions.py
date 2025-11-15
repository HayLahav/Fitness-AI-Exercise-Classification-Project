"""
Custom loss functions for imbalanced classification
"""

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def focal_loss(alpha: float = 1.0, gamma: float = 2.0):
    """
    Focal Loss implementation for imbalanced classification

    Focuses on hard examples by down-weighting easy examples.
    Useful for handling class imbalance in exercise classification.

    The focal loss was introduced in: https://arxiv.org/abs/1708.02002
    "Focal Loss for Dense Object Detection" by Lin et al.

    Args:
        alpha: Weighting factor for rare class (default: 1.0)
               Controls the balance between positive/negative examples
        gamma: Focusing parameter for hard examples (default: 2.0)
               Higher gamma = more focus on hard examples
               gamma=0 reduces to standard cross-entropy

    Returns:
        function: Focal loss function compatible with Keras

    Example:
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss=focal_loss(alpha=1.0, gamma=2.0),
        ...     metrics=['accuracy']
        ... )
    """

    def focal_loss_fn(y_true, y_pred):
        """
        Compute focal loss

        Args:
            y_true: Ground truth labels (sparse or one-hot encoded)
            y_pred: Predicted probabilities from softmax

        Returns:
            Scalar focal loss value
        """
        epsilon = tf.keras.backend.epsilon()

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Convert sparse labels to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)

        # Handle both sparse and one-hot encoded labels
        if len(y_true.shape) < len(y_pred.shape):
            y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        else:
            y_true_one_hot = tf.cast(y_true, tf.float32)

        # Calculate cross entropy
        ce = -y_true_one_hot * tf.math.log(y_pred)

        # Calculate focal weight: (1 - p_t)^gamma
        # where p_t is the probability of the true class
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow((1.0 - p_t), gamma)

        # Apply alpha weighting
        alpha_t = alpha * y_true_one_hot

        # Compute final focal loss
        fl = alpha_t * focal_weight * ce

        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    # Set function name for model saving/loading
    focal_loss_fn.__name__ = f'focal_loss_alpha_{alpha}_gamma_{gamma}'

    logger.debug(f"Focal loss created with alpha={alpha}, gamma={gamma}")

    return focal_loss_fn
