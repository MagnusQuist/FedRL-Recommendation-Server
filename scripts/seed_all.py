"""Standalone script to fully seed a fresh database (tables + backbones + catalogue).

Usage:
    DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db python scripts/seed_all.py

Or with local .env:
    python scripts/seed_all.py

Intended for first-time provisioning of a new RDS instance.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.db.db_init import ensure_models, ensure_seed_backbones
from app.db.seed_catalogue import seed_catalogue


async def main() -> None:
    print("=== Creating tables (if missing) ===")
    await ensure_models()

    print("\n=== Seeding backbone weights ===")
    await ensure_seed_backbones()

    print("\n=== Seeding food catalogue ===")
    await seed_catalogue()

    print("\n=== All done ===")


if __name__ == "__main__":
    asyncio.run(main())
