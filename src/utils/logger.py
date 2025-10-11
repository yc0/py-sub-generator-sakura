"""Logging utilities for the application."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "sakura_subtitle",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Setup application logger with file and console handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler (max 10MB, keep 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.warning(f"Failed to create file handler: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )


# Suppress noisy third-party loggers
def suppress_noisy_loggers():
    """Suppress verbose logging from third-party libraries."""
    noisy_loggers = [
        "transformers",
        "torch",
        "urllib3",
        "requests",
        "matplotlib",
        "PIL",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
