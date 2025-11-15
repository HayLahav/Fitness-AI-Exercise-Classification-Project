"""
Logging configuration for the application
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_string: Optional custom format string

    Returns:
        logging.Logger: Configured root logger

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="fitness_ai.log")
        >>> logger.info("Application started")
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True
    )

    logger = logging.getLogger()
    logger.info(f"Logging configured at {level} level")

    if log_file:
        logger.info(f"Logging to file: {log_file}")

    return logger
