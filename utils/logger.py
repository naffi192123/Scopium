"""
utils/logger.py

Structured logger that writes to both console and a rotating file.
Supports multi-process workers: each worker appends to the same shared log
file with its GPU rank prefixed in its logger name.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(
    name_or_path,
    results_dir=None,
    level=logging.INFO,
    worker_id: int = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
):
    """
    Set up a structured logger writing to console and optionally a log file.

    Can be called in two ways:

    **Legacy / main-process** (results_dir provided):
        setup_logger("wsi_framework", results_dir="results/")
        → log file: results/wsi_framework.log

    **Worker-process mode** (log_path provided as first argument):
        setup_logger("results/wsi_framework.log", worker_id=2)
        → appends to the shared rotating file with logger name "worker-2"

    Parameters
    ----------
    name_or_path : str   Logger name OR absolute path to a .log file.
    results_dir  : str   If given, creates ``results_dir/wsi_framework.log``.
    level        : int   Logging level (default INFO).
    worker_id    : int   If set, prefixes logger name with ``worker-{id}``.
    max_bytes    : int   Rotating file max size (default 10 MB).
    backup_count : int   Number of rotating backups to keep.

    Returns
    -------
    logging.Logger
    """
    # Resolve logger name and file path
    if results_dir is not None:
        # Main-process mode: derive path from results_dir
        log_file    = os.path.join(results_dir, 'wsi_framework.log')
        logger_name = 'wsi_framework'
    elif name_or_path.endswith('.log') or os.sep in name_or_path or '/' in name_or_path:
        # Worker-process mode: first arg IS the log file path
        log_file    = name_or_path
        logger_name = f'worker-{worker_id}' if worker_id is not None else 'worker'
    else:
        # Plain name, no file
        log_file    = None
        logger_name = name_or_path

    if worker_id is not None and not logger_name.startswith('worker'):
        logger_name = f'{logger_name}-gpu{worker_id}'

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Rotating file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
