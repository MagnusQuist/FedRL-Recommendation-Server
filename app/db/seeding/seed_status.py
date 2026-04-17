"""Helpers for inspecting database state.

Intentionally minimal — avoids importing the API layer to prevent circular
import cycles.
"""

from sqlalchemy import inspect, text

from app.db import AsyncSessionLocal, engine


_SENTINEL_TABLE = "categories"


async def has_tables() -> bool:
    """Return True if the server's schema is already present in the database.

    Uses the ``categories`` table as a sentinel — if it exists we assume the
    full schema has been created by a previous bootstrap or migration.
    """
    async with engine.begin() as conn:
        return await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).has_table(_SENTINEL_TABLE)
        )


async def is_database_seeded() -> bool:
    """Return True if the database already contains seed data."""
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                text(f"SELECT 1 FROM {_SENTINEL_TABLE} LIMIT 1")
            )
            return result.first() is not None
        except Exception:
            return False
