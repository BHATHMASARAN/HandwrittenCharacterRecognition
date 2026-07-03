import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config_loader import get_config


def get_logger(name: str) -> logging.Logger:
    config = get_config()
    log_dir = config.resolve_path(config.get("logging.log_dir", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    level_name = config.get("logging.level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
