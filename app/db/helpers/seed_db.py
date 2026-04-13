import os

from app.db.db_init import ensure_models, ensure_seed_catalogue
from app.logger import logger

from app.db.seed_status import is_database_seeded


def _auto_create_models_enabled() -> bool:
    return os.getenv("AUTO_CREATE_MODELS", "false").strip().lower() == "true"


async def ensure_models_if_enabled() -> None:
    """Create missing tables before serving when AUTO_CREATE_MODELS=true."""
    if _auto_create_models_enabled():
        await ensure_models()
        logger.info("Model creation phase done.")
    else:
        logger.info("Skipping create_all (AUTO_CREATE_MODELS=false).")


async def seed_data_if_needed() -> None:
    """Catalogue seed when the DB has no categories yet (can be slow).

    Backbone seeding is intentionally excluded here — it runs synchronously
    in the app lifespan before the CentralizedService is initialised.
    """
    if await is_database_seeded():
        logger.info("Existing data detected; skipping catalogue seed.")
        return

    await ensure_seed_catalogue()
    logger.info("Catalogue seeding done.")


async def seed_db() -> None:
    """Ensure models (if enabled) then seed data — for manual/CLI use; blocks until complete."""
    try:
        await ensure_models_if_enabled()
        await seed_data_if_needed()
    except Exception:
        logger.exception("seed_db failed.")
        raise
