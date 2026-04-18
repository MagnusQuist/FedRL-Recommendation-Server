"""API package.

This package houses the API layer of the application, including:
- request/response schemas (Pydantic)
- route definitions for FastAPI

This package is designed to be the single entrypoint for application code that
needs API schemas or routers.
"""

# Re-export subpackages for convenience (explicit aliases satisfy re-export lint rules).
from . import routers as routers
from . import schemas as schemas

__all__ = ["routers", "schemas"]
