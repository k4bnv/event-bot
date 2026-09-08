"""Central logging setup. One rotating file + a quiet console handler.

The console dashboard owns the terminal, so console logging is kept at
WARNING+ by default; everything (INFO/DEBUG included) always goes to
data/bot.log for post-mortem analysis.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(data_dir: Path, console_level: int = logging.WARNING) -> logging.Logger:
    logger = logging.getLogger("okx_event_bot")
    if logger.handlers:  # idempotent
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        data_dir / "bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger
