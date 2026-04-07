from app.api.schemas.catalogue_snapshot import CatalogueSnapshotResponse
from app.api.schemas.catalogue_version import CatalogueVersionResponse
from app.api.services.catalogue_snapshot import build_catalogue_snapshot, get_version
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db


router = APIRouter(prefix="/catalogue")

@router.get("/snapshot", response_model=CatalogueSnapshotResponse)
async def get_catalogue_snapshot(
    db: AsyncSession = Depends(get_db)
) -> CatalogueSnapshotResponse:
    return await build_catalogue_snapshot(db)

@router.get('/version', response_model=CatalogueVersionResponse)
async def get_catalogue_version(
    db: AsyncSession = Depends(get_db)
) -> CatalogueVersionResponse:
    return await get_version(db)