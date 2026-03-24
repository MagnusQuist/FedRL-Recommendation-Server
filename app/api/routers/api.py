"""API root router.

This module defines the versioned API root and wires up the individual
subrouters for catalogue, backbone, health, and developer metrics.

All endpoints are mounted under `/api/v1`.
"""

from fastapi import APIRouter
from app.api.routers import catalogue, backbone, health, metrics


router = APIRouter(prefix="/api/v1")

# Main API endpoints
router.include_router(health.router, tags=["Health"])
router.include_router(catalogue.router, tags=["Catalogue"])
router.include_router(backbone.router, tags=["Backbone"])

# Developer-only endpoints
router.include_router(metrics.router, tags=["Development"])