"""
Logging configuration for Horangi.

Usage:
    from core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting evaluation...")
    logger.error("Failed to load config")
"""

import logging
import sys
from typing import Optional


# Custom formatter with emoji support
class HorangiFormatter(logging.Formatter):
    """Custom formatter with colored output and emoji indicators."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    EMOJI = {
        logging.DEBUG: "",
        logging.INFO: "",
        logging.WARNING: "",
        logging.ERROR: "",
        logging.CRITICAL: "",
    }

    def __init__(self, use_colors: bool = True, use_emoji: bool = True):
        super().__init__()
        self.use_colors = use_colors
        self.use_emoji = use_emoji

    def format(self, record: logging.LogRecord) -> str:
        # Add emoji prefix
        emoji = self.EMOJI.get(record.levelno, "") if self.use_emoji else ""

        # Format message
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            msg = f"{color}{emoji}{record.getMessage()}{self.RESET}"
        else:
            msg = f"{emoji}{record.getMessage()}"

        return msg


# Global logger cache
_loggers: dict[str, logging.Logger] = {}
_configured = False


def configure_logging(
    level: int = logging.INFO,
    use_colors: bool = True,
    use_emoji: bool = True,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (default: INFO)
        use_colors: Whether to use colored output
        use_emoji: Whether to use emoji indicators
    """
    global _configured

    # Configure root logger for horangi
    root_logger = logging.getLogger("horangi")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(HorangiFormatter(use_colors=use_colors, use_emoji=use_emoji))
    root_logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    global _configured

    # Auto-configure with defaults if not already configured
    if not _configured:
        configure_logging()

    # Ensure logger is under horangi namespace
    if not name.startswith("horangi"):
        name = f"horangi.{name}"

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


# Convenience functions for logging without getting a logger first
def info(msg: str) -> None:
    """Log an info message."""
    get_logger("horangi").info(msg)


def warning(msg: str) -> None:
    """Log a warning message."""
    get_logger("horangi").warning(msg)


def error(msg: str) -> None:
    """Log an error message."""
    get_logger("horangi").error(msg)


def debug(msg: str) -> None:
    """Log a debug message."""
    get_logger("horangi").debug(msg)
