import logging
import os
import sys

LOGGER_NAME = "FedRL | Server"

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s %(levelname)s [%(name)s -> Worker pid: {os.getpid()}] %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False