"""API application factory.

This module provides a single entrypoint for building the FastAPI app.

Call `create_app()` to get a fully-configured `FastAPI` instance, or import
`app` directly (used by ASGI servers).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging import logger
from app.ml.aggregation import FLAggregator
from app.ml.centralized_training import CentralizedService
from app.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    aggregator = FLAggregator()
    await aggregator.try_load_persisted_state()
    app.state.aggregator = aggregator

    centralized_service = CentralizedService()
    await centralized_service.try_load_persisted_state()
    app.state.centralized_service = centralized_service

    logger.info("Server ready.")
    yield


def create_app() -> FastAPI:
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
    return app


app = create_app()
