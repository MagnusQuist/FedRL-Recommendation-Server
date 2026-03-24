from app.logger import logger

logger.info("STARTING APPLICATION")

from app.api.app import app  # noqa: F401
