import logging

logger = logging.getLogger("mlleakcatcher")

LEVELS_MAP = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

FORMATTER_MAP = {
    0: logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s",
        "%H:%M:%S",
    ),
    1: logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s",
        "%H:%M:%S",
    ),
    2: logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(funcName)s - %(levelname)s - %(message)s",
        "%d-%m-%Y %H:%M:%S",
    ),
}


def setup_logging(verbosity: int = 2) -> None:
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    level = LEVELS_MAP.get(verbosity, logging.INFO)
    formatter = FORMATTER_MAP.get(verbosity)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)



def set_verbosity(verbosity: int) -> None:

    level = LEVELS_MAP.get(verbosity, logging.INFO)
    formatter = FORMATTER_MAP.get(verbosity)

    for handler in logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    logger.warning(
        f"Logging level set to {logging.getLevelName(level)} (verbosity={verbosity})"
    )
