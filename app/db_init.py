import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.seed_backbone import seed_algorithm
from app.seed_catalogue import seed_catalogue


logger = logging.getLogger(__name__)


def _alembic_config() -> Config:
    """
    Create an Alembic Config pointing at this project's alembic.ini.

    This is resolved relative to the project root so it works both locally
    and inside Docker, assuming the working directory is the repo root.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(project_root / "alembic.ini"))
    return cfg


async def run_migrations() -> None:
    """
    Bring the database schema up to date using Alembic.

    Runs `alembic upgrade head` in a thread so it does not block the event loop.
    Safe to call on every startup: if already at head, it is a no-op.
    """
    logger.info("Running Alembic migrations to upgrade database to head...")
    cfg = _alembic_config()
    await asyncio.to_thread(command.upgrade, cfg, "head")
    logger.info("Alembic migrations complete.")


async def ensure_seed_backbones(algorithms: list[str] | None = None) -> None:
    """
    Ensure a version 0 backbone exists for the given algorithms.

    This is idempotent: if version 0 (or any version) already exists for
    an algorithm, the underlying seeding logic will skip creating a new one.
    """
    algos = algorithms or ["ts"]
    for algo in algos:
        logger.info("Ensuring version 0 backbone exists for algorithm='%s'...", algo)
        await seed_algorithm(algo)


async def ensure_seed_catalogue() -> None:
    """
    Ensure the food catalogue is seeded.

    The current implementation of `seed_catalogue` is effectively idempotent
    only when run on a fresh database, as it unconditionally inserts all rows.
    We call it only on startup so that a newly created volume is populated.
    If you need strict idempotency across restarts, extend `seed_catalogue`
    to check for existing rows before inserting.
    """
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()

