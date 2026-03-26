import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger():
    logger = logging.getLogger("kg_extractor")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "kg_extractor.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
    return logger

logger = setup_logger()
