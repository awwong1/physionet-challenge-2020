import logging


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_console_format = "[%(levelname)s] %(asctime)s: %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))

    logger.addHandler(console_handler)

    return logger
