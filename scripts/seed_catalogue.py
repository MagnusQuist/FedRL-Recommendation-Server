"""Standalone script to populate the food catalogue tables in PostgreSQL.

Usage:
    DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db python scripts/seed_catalogue.py

Or with local .env:
    python scripts/seed_catalogue.py

Requires that tables already exist (run Alembic migrations or set AUTO_CREATE_MODELS=true
on the first server boot).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.db.db_init import ensure_models
from app.db.seed_catalogue import seed_catalogue
from app.db.seed_status import is_database_seeded


async def main() -> None:
    await ensure_models()

    if await is_database_seeded():
        print("Database already contains catalogue data — re-seeding (full replace).")

    await seed_catalogue()


if __name__ == "__main__":
    asyncio.run(main())
