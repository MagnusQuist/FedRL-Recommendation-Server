import asyncio
from app.logger import logger
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.db.seed_backbone import DEFAULT_ALGORITHMS, seed_algorithms
from app.db.seed_catalogue import seed_catalogue


def _alembic_config() -> Config:
    """Create an Alembic Config pointing at this project's alembic.ini.

    This is resolved relative to the repository root (not the app package)
    so it works both locally and inside Docker.
    """
    project_root = Path(__file__).resolve().parents[2]
    return Config(str(project_root / "alembic.ini"))


async def run_migrations() -> None:
    logger.info("Running Alembic migrations to upgrade database to head...")
    cfg = _alembic_config()

    try:
        await asyncio.to_thread(command.upgrade, cfg, "head")
        logger.info("Alembic migrations complete.")
    except Exception:
        logger.exception("Alembic upgrade failed.")
        raise


async def ensure_seed_backbones(algorithms: list[str] | None = None) -> None:
    """Ensure the initial backbone (version 1) exists for the requested algorithms."""
    algos = algorithms or list(DEFAULT_ALGORITHMS)
    await seed_algorithms(algos)


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
