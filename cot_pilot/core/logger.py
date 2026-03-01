import logging
import os
import sys

def setup_logger(work_dir: str = "./workspace", log_file: str = "experiment.log") -> logging.Logger:
    """
    Sets up a logger that writes to both console and a file.
    
    Args:
        work_dir: Directory where the log file will be saved.
        log_file: Name of the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, log_file)
    
    logger = logging.getLogger("CoT-Pilot")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler - Detailed logs
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # File captures everything
    
    # Console Handler - User friendly logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # Simple format for console
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger() -> logging.Logger:
    """Returns the global logger instance."""
    return logging.getLogger("CoT-Pilot")
