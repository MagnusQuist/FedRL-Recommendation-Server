from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.api.schemas.category import Category
from app.api.schemas.food_item import FoodItem
from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup
from app.api.schemas.substitution_group import SubstitutionGroup

from app.api.models.food_item import (
    CatalogueSnapshot,
    FoodItemRead,
)
from app.api.models.substitution_group import SubstitutionGroups

router = APIRouter(prefix="/catalogue")


def _primary_substitution_group_id(item: FoodItem) -> int | None:
    if not item.substitution_groups:
        return None
    rel = min(item.substitution_groups, key=lambda g: g.group_priority)
    return rel.substitution_group_id


def food_item_read_from_orm(item: FoodItem) -> FoodItemRead:
    read = FoodItemRead.model_validate(item)
    gid = _primary_substitution_group_id(item)
    if gid is None:
        return read
    return read.model_copy(update={"substitution_group": gid})


def _catalogue_version(items: list[FoodItem]) -> str:
    if items:
        last_updated = max(item.created_at for item in items)
    else:
        last_updated = datetime.now(timezone.utc)

    return last_updated.astimezone(timezone.utc).isoformat()

@router.get("/version")
async def get_catalogue_version(db: AsyncSession = Depends(get_db)) -> dict:
    """Return a lightweight catalogue version object."""
    stmt = select(FoodItem)
    result = await db.execute(stmt)
    items: list[FoodItem] = result.scalars().unique().all()

    return {
        "version": _catalogue_version(items),
        "item_count": len(items),
    }


@router.get("/snapshot", response_model=CatalogueSnapshot)
async def get_catalogue_snapshot(db: AsyncSession = Depends(get_db)) -> CatalogueSnapshot:
    """Retrieve the catalogue snapshot."""

    # Items
    stmt_items = (
        select(FoodItem)
        .options(
            selectinload(FoodItem.category),
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            ),
        )
        .order_by(FoodItem.created_at)
    )
    result_items = await db.execute(stmt_items)
    items: list[FoodItem] = result_items.scalars().unique().all()
    payload: list[FoodItemRead] = [food_item_read_from_orm(item) for item in items]

    # Categories and substitution groups
    categories_result = await db.execute(select(Category))
    categories: list[Category] = categories_result.scalars().unique().all()

    substitution_groups_stmt = (
        select(SubstitutionGroup)
        .options(
            selectinload(SubstitutionGroup.food_items)
            .selectinload(FoodItemSubstitutionGroup.food_item)
        )
        .order_by(SubstitutionGroup.id)
    )

    substitution_groups_result = await db.execute(substitution_groups_stmt)
    substitution_groups: list[SubstitutionGroup] = substitution_groups_result.scalars().unique().all()

    substitution_groups_payload: list[SubstitutionGroups] = [
        SubstitutionGroups(id=group.id, name=group.name, item_ids=[rel.food_item.id for rel in group.food_items])
        for group in substitution_groups
    ]

    return CatalogueSnapshot(
        version=_catalogue_version(items),
        categories=categories,
        substitution_groups=substitution_groups_payload,
        items=payload,
    )