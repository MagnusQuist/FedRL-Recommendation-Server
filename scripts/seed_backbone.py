"""Standalone script to seed the initial backbone weights for both FL arms.

Usage:
    DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db python scripts/seed_backbone.py

Or with local .env:
    python scripts/seed_backbone.py

Idempotent — skips if rows already exist.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.db.db_init import ensure_models, ensure_seed_backbones


async def main() -> None:
    await ensure_models()
    await ensure_seed_backbones()


if __name__ == "__main__":
    asyncio.run(main())
