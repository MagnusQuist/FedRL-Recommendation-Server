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


class SuppressAccessPathFilter(logging.Filter):
    """Drop Uvicorn access logs for noisy endpoint prefixes."""

    _SUPPRESSED_PREFIXES = ("/api/v1/images",)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True

        path = self._extract_path(record.args)
        if path is None:
            return True

        return not any(path.startswith(prefix) for prefix in self._SUPPRESSED_PREFIXES)

    @staticmethod
    def _extract_path(args: object) -> str | None:
        if not isinstance(args, tuple):
            return None

        # Uvicorn access args are typically:
        # (client_addr, method, path, http_version, status_code)
        if len(args) >= 3 and isinstance(args[2], str):
            return args[2]

        # Fallback if record has a combined request-line format.
        if len(args) >= 2 and isinstance(args[1], str):
            request_line = args[1]
            parts = request_line.split(" ")
            if len(parts) >= 2:
                return parts[1]

        return None


def configure_uvicorn_access_filters() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, SuppressAccessPathFilter) for f in access_logger.filters):
        access_logger.addFilter(SuppressAccessPathFilter())


configure_uvicorn_access_filters()