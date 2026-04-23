"""Lightweight DB state checks (no API imports, avoids circular deps)."""

from sqlalchemy import inspect, text

from app.db import AsyncSessionLocal, engine


_SENTINEL_TABLE = "categories"


async def has_table(table_name: str) -> bool:
    """True if ``table_name`` exists."""
    async with engine.begin() as conn:
        return await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).has_table(table_name)
        )


async def has_tables() -> bool:
    """True if ``categories`` exists (sentinel for schema present)."""
    return await has_table(_SENTINEL_TABLE)


async def has_column(table_name: str, column_name: str) -> bool:
    """True if ``table_name`` contains ``column_name``."""
    async with engine.begin() as conn:
        return await conn.run_sync(
            lambda sync_conn: (
                inspect(sync_conn).has_table(table_name)
                and any(
                    col.get("name") == column_name
                    for col in inspect(sync_conn).get_columns(table_name)
                )
            )
        )


async def is_database_seeded() -> bool:
    """True if ``categories`` has at least one row."""
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                text(f"SELECT 1 FROM {_SENTINEL_TABLE} LIMIT 1")
            )
            return result.first() is not None
        except Exception:
            return False
