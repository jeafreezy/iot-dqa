import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    "iot-dqa:%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def add_file_logging(file_path="iot-dqa.log"):
    """
    Add file logging to the logger.

    Args:
        file_path (str): Path to the log file. Defaults to 'iot-dqa.log'.
    """
    file_handler = logging.FileHandler(file_path)
    file_formatter = logging.Formatter(
        "iot-dqa:%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def configure_logging(level=logging.WARNING):
    """
    Configure the logging level for the package.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)

    Example:
        # Set logging level to INFO
        configure_logging(logging.INFO)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
