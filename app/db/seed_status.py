"""Helpers for checking whether the database has been seeded.

This module is intentionally minimal and avoids importing the API layer to
prevent circular import cycles.
"""

from sqlalchemy import text

from app.db import AsyncSessionLocal


async def is_database_seeded() -> bool:
    """Return True if the database already contains seed data."""
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(text("SELECT 1 FROM categories LIMIT 1"))
            return result.first() is not None
        except Exception:
            # Schema might not exist yet (tables not created), or DB not ready.
            return False
