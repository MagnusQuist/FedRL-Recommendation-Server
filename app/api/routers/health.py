from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.seeding.seed_status import is_database_seeded

router = APIRouter()


@router.get("/health", summary="Health check")
async def health(db: AsyncSession = Depends(get_db)):
    """
    Returns server and database status.
    Used by Raspberry Pi clients to detect connectivity before attempting a sync.
    """
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "reachable"}


@router.get("/seed-status", summary="Database seed status")
async def seed_status():
    """Returns whether the database has already been seeded."""
    seeded = await is_database_seeded()
    return {"seeded": seeded}