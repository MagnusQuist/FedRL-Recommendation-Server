import os

from app.db.db_init import ensure_seed_backbones, ensure_seed_catalogue, run_migrations
from app.logger import logger

from app.db.seed_status import is_database_seeded

async def seed_db() -> None:
    try:
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
    except Exception:
        logger.exception("seed_db failed.")
        raise