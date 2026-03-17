import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.db.seed_backbone import seed_algorithm
from app.db.seed_catalogue import seed_catalogue


logger = logging.getLogger(__name__)


def _alembic_config() -> Config:
    """Create an Alembic Config pointing at this project's alembic.ini.

    This is resolved relative to the repository root (not the app package)
    so it works both locally and inside Docker.
    """
    project_root = Path(__file__).resolve().parents[2]
    return Config(str(project_root / "alembic.ini"))


async def run_migrations() -> None:
    """Bring the database schema up to date using Alembic."""
    logger.info("Running Alembic migrations to upgrade database to head...")
    cfg = _alembic_config()
    await asyncio.to_thread(command.upgrade, cfg, "head")
    logger.info("Alembic migrations complete.")


async def ensure_seed_backbones(algorithms: list[str] | None = None) -> None:
    """Ensure a version 0 backbone exists for the given algorithms."""
    algos = algorithms or ["ts", "dqn"]
    for algo in algos:
        logger.info("Ensuring version 0 backbone exists for algorithm='%s'...", algo)
        await seed_algorithm(algo)


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
