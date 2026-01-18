"""
Logging Configuration

Centralized logging using loguru with structured output.
"""

import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add console handler with custom format
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

# Add file handler for persistent logs
logger.add(
    "logs/insights_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
)


def get_logger(name: str):
    """Get a logger with a specific name for component identification."""
    return logger.bind(name=name)


# Pre-configured loggers for different components
upload_logger = get_logger("upload")
insights_logger = get_logger("insights")
llm_logger = get_logger("llm")
data_logger = get_logger("data")
cache_logger = get_logger("cache")
