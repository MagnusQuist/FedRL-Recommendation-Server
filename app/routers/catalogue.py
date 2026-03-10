from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.food_item import FoodItem
from app.models.food_item import CatalogueResponse, CategoryResponse, FoodItemRead


router = APIRouter()


@router.get(
    "",
    response_model=CatalogueResponse,
    summary="Full food catalogue",
)
async def get_catalogue(db: AsyncSession = Depends(get_db)) -> CatalogueResponse:
    result = await db.execute(select(FoodItem).order_by(FoodItem.created_at))
    items = list(result.scalars().all())

    if items:
        last_updated = max(item.created_at for item in items)
    else:
        last_updated = datetime.now(timezone.utc)

    version = last_updated.astimezone(timezone.utc).isoformat()

    return CatalogueResponse(
        version=version,
        item_count=len(items),
        items=items,
    )


@router.get(
    "/category/{name}",
    response_model=CategoryResponse,
    summary="Items filtered by category",
)
async def get_category(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    result = await db.execute(
        select(FoodItem).where(FoodItem.category == name).order_by(FoodItem.created_at)
    )
    items = list(result.scalars().all())

    if items:
        last_updated = max(item.created_at for item in items)
    else:
        last_updated = datetime.now(timezone.utc)

    version = last_updated.astimezone(timezone.utc).isoformat()

    return CategoryResponse(
        version=version,
        category=name,
        item_count=len(items),
        items=items,
    )


@router.get(
    "/{item_id}",
    response_model=FoodItemRead,
    summary="Single item by UUID",
)
async def get_item(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FoodItemRead:
    result = await db.execute(
        select(FoodItem).where(FoodItem.id == item_id)
    )
    item = result.scalars().first()

    if item is None:
        raise HTTPException(status_code=404, detail="Food item not found")

    return item

