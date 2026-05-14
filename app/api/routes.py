from fastapi import APIRouter

from app.api.endpoints import catalogue, centralized, dev, federated, health, images

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["Health"])
router.include_router(catalogue.router, tags=["Catalogue"])
router.include_router(images.router, tags=["Images"])
router.include_router(federated.router, tags=["Federated"])
router.include_router(centralized.router, tags=["Centralized"])
router.include_router(dev.router, tags=["Development"])
