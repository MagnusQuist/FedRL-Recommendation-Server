from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routers import catalogue, health, fl


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context.

    In development you can still run `Base.metadata.create_all()` manually or via
    Alembic migrations. We avoid creating tables automatically here to prevent
    conflicts with Alembic's schema management.
    """
    yield


app = FastAPI(
    title="FedRL Recommendation Server",
    description=(
        "Federated RL recommendation server for the Nudge2Green project. "
        "Exposes the food catalogue API and the federated learning aggregation endpoints."
    ),
    version="0.1.0",
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