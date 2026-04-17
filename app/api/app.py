"""API application factory.

This module provides a single entrypoint for building the FastAPI app.

Call `create_app()` to get a fully-configured `FastAPI` instance, or import
`app` directly (used by ASGI servers).
"""

from app.logger import logger

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db.db_init import ensure_models
from app.db import AsyncSessionLocal
from app.backbones.aggregator import FLAggregator, ROUND_TIMEOUT_SECONDS
from app.backbones.centralized import CentralizedService
from app.api.routers.api import router as api_router


async def _timeout_watcher(aggregator: FLAggregator) -> None:
    """Background task that periodically checks for round timeouts."""
    logger.info("FL timeout watcher started (checking every 10s, timeout=%ds).", ROUND_TIMEOUT_SECONDS)
    while True:
        await asyncio.sleep(10)
        try:
            async with AsyncSessionLocal() as db:
                triggered = await aggregator.check_timeout(db)
                if triggered:
                    logger.info("Timeout watcher triggered a FedAvg round.")
        except Exception:
            logger.exception("Error in FL timeout watcher.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. Create missing DB tables (if AUTO_CREATE_MODELS=true).
      2. Load persisted state for FL services.
      3. Start the FL timeout watcher.

    Database seeding (backbones + catalogue) is handled by standalone scripts
    in scripts/ and should be run before the first deployment.
    """
    logger.info("Starting DB init...")
    try:
        await ensure_models(check_env=True)
    except Exception:
        logger.exception("Startup failed during model creation.")
        raise

    aggregator = FLAggregator()
    app.state.aggregator = aggregator

    centralized_service = CentralizedService()
    await centralized_service.try_load_persisted_state()
    app.state.centralized_service = centralized_service

    watcher = asyncio.create_task(_timeout_watcher(aggregator))

    logger.info("Server ready.")

    yield

    watcher.cancel()
    try:
        await watcher
    except asyncio.CancelledError:
        pass


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
