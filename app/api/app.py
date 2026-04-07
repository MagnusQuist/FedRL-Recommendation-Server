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

from app.api.helpers.seed_db import ensure_models_if_enabled, seed_data_if_needed
from app.db import AsyncSessionLocal
from app.fl.aggregator import FLAggregator, ROUND_TIMEOUT_SECONDS
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
    """Model creation (create_all) runs before serving; heavy catalogue seed runs in the background."""
    logger.info("Starting DB init...")
    try:
        await ensure_models_if_enabled()
        logger.info("Model creation phase finished.")
    except Exception:
        logger.exception("Startup failed during model creation.")
        raise

    aggregator = FLAggregator()
    app.state.aggregator = aggregator

    watcher = asyncio.create_task(_timeout_watcher(aggregator))

    if os.getenv("AUTO_SEED_DATA_ON_STARTUP", "true").strip().lower() == "true":

        async def _background_seed() -> None:
            try:
                await seed_data_if_needed()
                logger.info("Background catalogue/backbone seed finished.")
            except Exception:
                logger.exception(
                    "Background seed failed — ensure models (AUTO_CREATE_MODELS=true) and "
                    "`python -m app.db.seed_catalogue` if the catalogue is empty."
                )

        asyncio.create_task(_background_seed())
        logger.info("Catalogue/backbone seed scheduled in background (server is starting).")
    else:
        logger.info("AUTO_SEED_DATA_ON_STARTUP=false — skipping automatic data seed.")

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
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
