import os

from app.db.db_init import ensure_seed_backbones, ensure_seed_catalogue, run_migrations
from fastapi import logger

from app.db.seed_status import is_database_seeded

logger = logger.getLogger(__name__)

async def seed_db() -> None:
    # Running migrations on every startup is convenient for development, but may
    # not be ideal for production. Control this via an env var so it can be
    # disabled in deployment pipelines where migrations are managed separately.
    if os.getenv("AUTO_RUN_MIGRATIONS", "false").strip().lower() == "true":
        await run_migrations()
        logger.info("Migrations done")
    else:
        logger.info("Skipping migrations (AUTO_RUN_MIGRATIONS=false)")

    if not await is_database_seeded():
        await ensure_seed_backbones()
        logger.info("Backbone seeding done")

        await ensure_seed_catalogue()
        logger.info("Catalogue seeding done")
    else:
        logger.info("Existing data detected; skipping seed data population.")