import logging

import colorlog

# Create a custom logger
logger = colorlog.getLogger()

# Set the level of logger
logger.setLevel(logging.DEBUG)

# Create a handler
handler = colorlog.StreamHandler()

handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
)

# Add handler to the logger
logger.addHandler(handler)
