"""API route modules.

Expose router modules so callers can do:

    from app.api.routers import catalogue, backbone, health, metrics

This mirrors the previous `app.routers` package layout.
"""

from app.api.routers import catalogue, backbone, health, metrics  # noqa: F401
