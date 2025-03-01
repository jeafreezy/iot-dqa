import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "iot-dqa:%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
