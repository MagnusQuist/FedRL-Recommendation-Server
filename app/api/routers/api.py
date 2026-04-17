"""API root router.

This module defines the versioned API root and wires up the individual
subrouters for catalogue, federated, centralized, health, and developer metrics.

All endpoints are mounted under `/api/v1`.
"""

from fastapi import APIRouter

from app.api.routers import catalogue, centralized, federated, health, images, metrics


router = APIRouter(prefix="/api/v1")

# Main API endpoints
router.include_router(health.router, tags=["Health"])
router.include_router(catalogue.router, tags=["Catalogue"])
router.include_router(images.router, tags=["Images"])
router.include_router(federated.router, tags=["Federated"])
router.include_router(centralized.router, tags=["Centralized"])

# Developer-only endpoints
router.include_router(metrics.router, tags=["Development"])
