"""Consistent timestamped console and file logging for the full project."""
import logging
from pathlib import Path


def configure_logging(log_file=None, level=logging.INFO):
    """Configure root logging once; log to console and optionally a UTF-8 file."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("vgg6")
