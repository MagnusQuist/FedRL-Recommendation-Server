import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import AsyncSessionLocal
from app.db_init import ensure_seed_backbones, ensure_seed_catalogue, run_migrations
from app.fl.aggregator import FLAggregator, ROUND_TIMEOUT_SECONDS
from app.routers import catalogue, health, fl, metrics

logger = logging.getLogger(__name__)

async def _timeout_watcher(aggregator: FLAggregator) -> None:
    """
    Background task that runs for the lifetime of the server.
    Wakes every 10 seconds and checks whether the round timeout has elapsed.
    If it has, and enough clients are queued, it triggers a FedAvg round.
    Uploads below min_clients_per_round are carried forward — never discarded.
    """
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
    # Bring database schema up to date and ensure initial seed data exists.
    # All operations are intended to be safe to call on every startup:
    # - Alembic migrations only apply pending changes.
    # - Backbone seeding only creates version 0 backbones if they don't exist yet.
    # - Catalogue seeding currently assumes a fresh database; extend it if
    #   you need strict idempotency across restarts with existing data.
    await run_migrations()
    await ensure_seed_backbones()
    await ensure_seed_catalogue()

    # Mount the aggregator singleton on app state so routers can access it
    aggregator = FLAggregator()
    app.state.aggregator = aggregator

    # Start the background timeout watcher
    watcher = asyncio.create_task(_timeout_watcher(aggregator))

    yield

    # Shutdown
    watcher.cancel()
    try:
        await watcher
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="FedRL Recommendation Server",
    description=(
        "Federated RL recommendation server for the Nudge2Green project. "
        "Exposes the food catalogue API and the federated learning aggregation endpoints."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(catalogue.router, prefix="/catalogue", tags=["Catalogue"])
app.include_router(fl.router, prefix="/fl", tags=["Federated Learning"])
app.include_router(metrics.router, prefix="/dev", tags=["Development"])

# Serve dev metrics static assets (CSS, JS, etc.) from app/web under /dev/static
DEV_WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount(
    "/dev/static",
    StaticFiles(directory=DEV_WEB_DIR),
    name="dev-static",
)