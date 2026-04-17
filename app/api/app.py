"""API application factory.

This module provides a single entrypoint for building the FastAPI app.

Call `create_app()` to get a fully-configured `FastAPI` instance, or import
`app` directly (used by ASGI servers).
"""

from app.logger import logger

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backbones.aggregator import FLAggregator
from app.backbones.centralized import CentralizedService
from app.api.routers.api import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. Initialise the FL aggregator singleton.
      2. Load persisted state for the centralized service.
    """
    aggregator = FLAggregator()
    app.state.aggregator = aggregator

    centralized_service = CentralizedService()
    await centralized_service.try_load_persisted_state()
    app.state.centralized_service = centralized_service

    logger.info("Server ready.")

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="FedRL Recommendation Server",
        description=(
            "Federated RL recommendation server for the Nudge2Green project. "
            "Exposes the food catalogue API and the federated learning aggregation endpoints."
        ),
        version="0.3.1",
        lifespan=lifespan,
    )

    allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve dev metrics static assets (CSS, JS, etc.) from app/web under /api/v1/dev/static
    DEV_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
    app.mount(
        "/api/v1/dev/static",
        StaticFiles(directory=DEV_WEB_DIR),
        name="dev-static",
    )

    return app


app = create_app()
