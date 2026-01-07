import logging
from loguru import logger
from pathlib import Path
import sys


def configure_logger(
    name: str = "zigzag_system",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True
):
    """
    Configure logging for the system
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Configure file logging
    if file:
        log_file = log_path / f"{name}.log"
        logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="500 MB",
            retention="7 days"
        )
    
    # Configure console logging
    if console:
        logger.add(
            sys.stdout,
            level=level,
            format="<level>{level: <8}</level> | <cyan>{name}:{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    
    return logger


def get_logger(name: str = __name__):
    """
    Get logger instance
    
    Args:
        name: Module name
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerConfig:
    """
    Logger configuration management
    """
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    _configured = False
    _logger = None
    
    @classmethod
    def setup(
        cls,
        level: str = INFO,
        log_dir: str = "logs",
        console: bool = True,
        file: bool = True
    ):
        """
        Setup logger globally
        
        Args:
            level: Logging level
            log_dir: Log directory
            console: Enable console output
            file: Enable file output
        """
        if not cls._configured:
            cls._logger = configure_logger(
                log_dir=log_dir,
                level=level,
                console=console,
                file=file
            )
            cls._configured = True
    
    @classmethod
    def get(cls):
        """
        Get configured logger
        
        Returns:
            Logger instance
        """
        if not cls._configured:
            cls.setup()
        return cls._logger


# Example log format templates
LOG_FORMAT_DETAILED = "<level>{level: <8}</level> | {time:YYYY-MM-DD HH:mm:ss} | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
LOG_FORMAT_SIMPLE = "<level>{level: <8}</level> | {time:HH:mm:ss} - <level>{message}</level>"
LOG_FORMAT_FILE = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


if __name__ == "__main__":
    # Test logging
    logger_instance = configure_logger(level="DEBUG")
    
    logger_instance.debug("This is a debug message")
    logger_instance.info("This is an info message")
    logger_instance.warning("This is a warning message")
    logger_instance.error("This is an error message")
    logger_instance.critical("This is a critical message")
