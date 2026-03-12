from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter()


@router.get("/health", summary="Health check")
async def health(db: AsyncSession = Depends(get_db)):
    """
    Returns server and database status.
    Used by Raspberry Pi clients to detect connectivity before attempting a sync.
    """
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "reachable"}