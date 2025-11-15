"""
Neural network model architecture for exercise classification
"""

import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate,
    MultiHeadAttention, LayerNormalization
)

from ..config import (
    LSTM_UNITS_1, LSTM_UNITS_2, ATTENTION_HEADS, ATTENTION_KEY_DIM,
    DENSE_UNITS_1, DENSE_UNITS_2, DROPOUT_RATE_1, DROPOUT_RATE_2
)

logger = logging.getLogger(__name__)


def create_attention_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units_1: int = LSTM_UNITS_1,
    lstm_units_2: int = LSTM_UNITS_2,
    attention_heads: int = ATTENTION_HEADS,
    attention_key_dim: int = ATTENTION_KEY_DIM,
    dense_units_1: int = DENSE_UNITS_1,
    dense_units_2: int = DENSE_UNITS_2,
    dropout_rate_1: float = DROPOUT_RATE_1,
    dropout_rate_2: float = DROPOUT_RATE_2
) -> Model:
    """
    Create attention-based BiLSTM model for exercise classification

    Architecture:
    1. Bidirectional LSTM layers for temporal feature learning
    2. Layer normalization for training stability
    3. Multi-head attention mechanism for temporal dependencies
    4. Global pooling (average + max) for feature aggregation
    5. Dense classification head with regularization

    This architecture is designed to:
    - Capture temporal dependencies in exercise movements
    - Focus on discriminative features through attention
    - Handle variable-length sequences effectively
    - Regularize to prevent overfitting on small datasets

    Args:
        input_shape: Shape of input sequences (timesteps, features)
                    e.g., (20, 83) for 20 frames with 83 features each
        num_classes: Number of exercise classes to classify
        lstm_units_1: Units in first LSTM layer (default from config)
        lstm_units_2: Units in second LSTM layer (default from config)
        attention_heads: Number of attention heads (default from config)
        attention_key_dim: Dimension of attention keys (default from config)
        dense_units_1: Units in first dense layer (default from config)
        dense_units_2: Units in second dense layer (default from config)
        dropout_rate_1: Dropout rate for first dense layer (default from config)
        dropout_rate_2: Dropout rate for LSTM layers (default from config)

    Returns:
        keras.Model: Compiled neural network model

    Example:
        >>> model = create_attention_model(
        ...     input_shape=(20, 83),
        ...     num_classes=22
        ... )
        >>> model.summary()
    """
    logger.info(f"Creating attention model with input shape {input_shape} "
                f"and {num_classes} classes")

    # Input layer
    inputs = Input(shape=input_shape, name='exercise_sequence_input')

    # First Bidirectional LSTM layer
    # Bidirectional processes sequence forward and backward for better context
    lstm1 = Bidirectional(
        LSTM(lstm_units_1, return_sequences=True, dropout=dropout_rate_2),
        name='bilstm_layer_1'
    )(inputs)
    lstm1 = LayerNormalization(name='layer_norm_1')(lstm1)

    # Second Bidirectional LSTM layer
    lstm2 = Bidirectional(
        LSTM(lstm_units_2, return_sequences=True, dropout=dropout_rate_2),
        name='bilstm_layer_2'
    )(lstm1)
    lstm2 = LayerNormalization(name='layer_norm_2')(lstm2)

    # Multi-head attention mechanism
    # Allows model to focus on different temporal aspects simultaneously
    attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_key_dim,
        name='multi_head_attention'
    )(lstm2, lstm2)
    attention = LayerNormalization(name='layer_norm_attention')(attention)

    # Global pooling layers
    # Aggregate sequence information into fixed-size representation
    avg_pool = GlobalAveragePooling1D(name='global_avg_pool')(attention)
    max_pool = GlobalMaxPooling1D(name='global_max_pool')(attention)

    # Combine pooled features
    combined = Concatenate(name='concatenate_pools')([avg_pool, max_pool])
    combined = BatchNormalization(name='batch_norm_combined')(combined)

    # Classification head
    # First dense layer with ReLU activation
    dense1 = Dense(dense_units_1, activation='relu', name='dense_1')(combined)
    dense1 = Dropout(dropout_rate_1, name='dropout_1')(dense1)
    dense1 = BatchNormalization(name='batch_norm_dense_1')(dense1)

    # Second dense layer
    dense2 = Dense(dense_units_2, activation='relu', name='dense_2')(dense1)
    dense2 = Dropout(dropout_rate_2, name='dropout_2')(dense2)

    # Output layer with softmax for multi-class classification
    outputs = Dense(num_classes, activation='softmax', name='exercise_output')(dense2)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='ExerciseClassifier')

    logger.info(f"Model created successfully with {model.count_params():,} parameters")

    return model


def create_simple_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 64
) -> Model:
    """
    Create a simpler LSTM model for comparison/baseline

    Args:
        input_shape: Shape of input sequences (timesteps, features)
        num_classes: Number of exercise classes
        lstm_units: Number of LSTM units

    Returns:
        keras.Model: Simple LSTM model
    """
    logger.info(f"Creating simple LSTM model with {lstm_units} units")

    inputs = Input(shape=input_shape)
    lstm = LSTM(lstm_units)(inputs)
    dense = Dense(32, activation='relu')(lstm)
    dropout = Dropout(0.3)(dense)
    outputs = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs, name='SimpleLSTM')

    logger.info(f"Simple model created with {model.count_params():,} parameters")

    return model
