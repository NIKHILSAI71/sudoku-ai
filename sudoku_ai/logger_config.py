"""
Logging configuration for Sudoku AI
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> Path | None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file

    Returns:
        Path to log file if logging to file, else None
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    log_file = None
    if log_to_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"sudoku_ai_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_file
