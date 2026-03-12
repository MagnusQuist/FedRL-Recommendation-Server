"""API application factory.

This module provides a single entrypoint for building the FastAPI app.

Call `create_app()` to get a fully-configured `FastAPI` instance, or import
`app` directly (used by ASGI servers).
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os

from app.db import AsyncSessionLocal
from app.db.db_init import ensure_seed_backbones, ensure_seed_catalogue, run_migrations
from app.db.seed_status import is_database_seeded
from app.fl.aggregator import FLAggregator, ROUND_TIMEOUT_SECONDS
from app.api.routers.api import router as api_router

logger = logging.getLogger(__name__)


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
    """Application lifespan context manager.

    This is used by FastAPI to run startup and shutdown logic.
    """
    logger.info("Starting DB init...")

    # Running migrations on every startup is convenient for development, but may
    # not be ideal for production. Control this via an env var so it can be
    # disabled in deployment pipelines where migrations are managed separately.
    if os.getenv("RUN_MIGRATIONS_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes"}:
        await run_migrations()
        logger.info("Migrations done")
    else:
        logger.info("Skipping migrations (RUN_MIGRATIONS_ON_STARTUP=false)")

    if not await is_database_seeded():
        await ensure_seed_backbones()
        logger.info("Backbone seeding done")

        await ensure_seed_catalogue()
        logger.info("Catalogue seeding done")
    else:
        logger.info("Existing data detected; skipping seed data population.")

    aggregator = FLAggregator()
    app.state.aggregator = aggregator

    watcher = asyncio.create_task(_timeout_watcher(aggregator))

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
        version="0.3.0",
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
