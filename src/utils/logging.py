# src/utils/logging.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. Assumes configure_logging() has already been called.
    Safe to call even if not configured (will inherit root handlers).
    """
    return logging.getLogger(name)


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "run.log",
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    console: bool = True,
) -> None:
    """
    Configure root logging exactly once.

    - Console handler (optional)
    - File handler (optional, if log_dir provided)

    This is intentionally conservative: no fancy async handlers, no external deps.
    """
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    root.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Avoid duplicate handlers if configure_logging is called multiple times
    # (common in notebooks).
    def _has_handler(handler_type: type) -> bool:
        return any(isinstance(h, handler_type) for h in root.handlers)

    if console and not _has_handler(logging.StreamHandler):
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(log_dir) / log_file

        # FileHandler is a subclass of StreamHandler, so the above check isn't enough.
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(file_path)
                   for h in root.handlers):
            fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            fh.setLevel(numeric_level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    # Reduce noise from common libraries unless user explicitly wants DEBUG.
    if numeric_level > logging.DEBUG:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)


def configure_from_env(
    *,
    default_level: str = "INFO",
    default_log_dir: Optional[str] = None,
) -> None:
    """
    Optional helper: configure logging from environment variables.

    Env vars:
      - LOG_LEVEL
      - LOG_DIR
      - LOG_FILE
    """
    level = os.getenv("LOG_LEVEL", default_level)
    log_dir = os.getenv("LOG_DIR", default_log_dir)
    log_file = os.getenv("LOG_FILE", "run.log")
    configure_logging(level=level, log_dir=log_dir, log_file=log_file)
