import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "LunoInstitutionalEngine") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "engine.log")
    
    log_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(log_handler)
        
    return logger
