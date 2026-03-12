"""API package.

This package houses the API layer of the application, including:
- request/response schemas (Pydantic)
- database models
- route definitions for FastAPI

This package is designed to be the single entrypoint for application code that
needs API models or routers.
"""

# Export subpackages at the package level for convenience.
from app.api import models, routers, schemas  # noqa: F401
