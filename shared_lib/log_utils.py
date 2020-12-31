import logging

""" 
log leverls
- logger.debug("debug message")
- logger.info("info message")
- logger.warning("warning message")
- logger.error("error message")
- logger.critical("critical message")
"""


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s : %(levelname)s : %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_logger(log_path, log_level=logging.INFO):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
        log_format: log or color
        log_level: lowest log levels, in the following order
            - logging.DEBUG
            - logging.INFO
            - logging.WARNING
            - logging.ERROR
            - logging.CRITICAL
            default is logging.INFO
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(CustomFormatter())
        logger.addHandler(stream_handler)
