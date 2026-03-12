import logging
import os
import sys
from datetime import datetime

def setup_logger(name, results_dir=None, level=logging.INFO):
    """
    Sets up a structured logger that outputs to both console and optionally a file.
    
    Args:
        name (str): The name of the logger (usually __name__).
        results_dir (str, optional): The root path to save logs. If provided,
                                     a .log file will be created in results_dir/logs/.
        level (int): The logging level.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        logger.handlers.clear()
        
    # Create formatter: [YYYY-MM-DD HH:MM:SS] [LEVEL] [LOGGER_NAME] Message
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # stdout Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # File Handler
    if results_dir:
        logs_dir = os.path.join(results_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a timestamped log file, e.g., run_20231024_153022.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"run_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Prevent basicConfig collisions
    logger.propagate = False
    
    return logger
