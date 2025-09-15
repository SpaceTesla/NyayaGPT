"""Logging configuration for NyayaGPT."""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "nyayagpt", level: int = logging.INFO) -> logging.Logger:
    """Set up logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


# Default logger
logger = setup_logger()
