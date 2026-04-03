from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.api.schemas.category import Category
from app.api.schemas.food_item import FoodItem
from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup
from app.api.schemas.substitution_group import SubstitutionGroup

from app.api.models.food_item import (
    CatalogueResponse,
    CategoryResponse,
    CatalogueSnapshot,
    FoodItemRead,
    SubstitutionGroupItemResponse,
)
from app.api.models.substitution_group_with_items import SubstitutionGroupWithItems

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


@router.get("", response_model=CatalogueResponse)
async def get_catalogue(db: AsyncSession = Depends(get_db)) -> CatalogueResponse:
    stmt = (
        select(FoodItem)
        .options(
            selectinload(FoodItem.category),
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            ),
        )
        .order_by(FoodItem.created_at)
    )

    result = await db.execute(stmt)
    items = list(result.scalars().unique().all())

    try:
        payload = [food_item_read_from_orm(item) for item in items]
    except Exception as exc:
        print("Catalogue serialization failed:", repr(exc))
        raise

    return CatalogueResponse(
        version=_catalogue_version(items),
        item_count=len(items),
        items=payload,
    )


@router.get("/version")
async def get_catalogue_version(db: AsyncSession = Depends(get_db)) -> dict:
    """Return a lightweight catalogue version object."""
    stmt = select(FoodItem)
    result = await db.execute(stmt)
    items = list(result.scalars().unique().all())

    return {
        "version": _catalogue_version(items),
        "item_count": len(items),
    }


@router.get("/snapshot", response_model=CatalogueSnapshot)
async def get_catalogue_snapshot(db: AsyncSession = Depends(get_db)) -> CatalogueSnapshot:
    """Return all catalogue content (items + taxonomy) in one response."""

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
    items = list(result_items.scalars().unique().all())
    payload = [food_item_read_from_orm(item) for item in items]

    # Categories and substitution groups
    categories_result = await db.execute(select(Category))
    categories = list(categories_result.scalars().unique().all())

    substitution_groups_stmt = (
        select(SubstitutionGroup)
        .options(
            selectinload(SubstitutionGroup.food_items)
            .selectinload(FoodItemSubstitutionGroup.food_item)
        )
        .order_by(SubstitutionGroup.id)
    )

    substitution_groups_result = await db.execute(substitution_groups_stmt)
    substitution_groups = list(substitution_groups_result.scalars().unique().all())

    substitution_groups_payload: list[SubstitutionGroupWithItems] = []
    for group in substitution_groups:
        item_ids = [rel.food_item.id for rel in group.food_items]
        substitution_groups_payload.append(
            SubstitutionGroupWithItems(group=group, items=item_ids)
        )

    return CatalogueSnapshot(
        version=_catalogue_version(items),
        categories=categories,
        substitution_groups=substitution_groups_payload,
        items=payload,
    )


@router.get("/substitution_groups", response_model=list[SubstitutionGroupWithItems])
async def get_substitution_groups(
    db: AsyncSession = Depends(get_db)
) -> list[SubstitutionGroupWithItems]:
    """Return all substitution groups with their related food items."""

    stmt = (
        select(SubstitutionGroup)
        .options(
            selectinload(SubstitutionGroup.food_items)
            .selectinload(FoodItemSubstitutionGroup.food_item)
            .selectinload(FoodItem.category),
        )
        .order_by(SubstitutionGroup.id)
    )

    result = await db.execute(stmt)
    groups = list(result.scalars().unique().all())

    payload: list[SubstitutionGroupWithItems] = []
    for group in groups:
        items = [rel.food_item.id for rel in group.food_items]
        payload.append(SubstitutionGroupWithItems(group=group, items=items))

    return payload


@router.get("/substitution_groups/item", response_model=SubstitutionGroupItemResponse)
async def get_item_substitution_group(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> SubstitutionGroupItemResponse:
    """Get the substitution group for an item plus related items in that group."""
    stmt = (
        select(FoodItem)
        .where(FoodItem.id == item_id)
        .options(
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            )
        )
    )

    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Food item not found")

    if not item.substitution_groups:
        raise HTTPException(
            status_code=404,
            detail="No substitution group found for this item.",
        )

    # Select the highest-priority substitution group if multiple exist
    group_rel = min(item.substitution_groups, key=lambda g: g.group_priority)
    group = group_rel.substitution_group

    # Find other items in the same group
    stmt_related = (
        select(FoodItem)
        .join(FoodItemSubstitutionGroup, FoodItem.id == FoodItemSubstitutionGroup.food_item_id)
        .where(FoodItemSubstitutionGroup.substitution_group_id == group.id)
        .options(
            selectinload(FoodItem.category),
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            ),
        )
        .order_by(FoodItem.created_at)
    )

    result_related = await db.execute(stmt_related)
    related_items = list(result_related.scalars().unique().all())

    return SubstitutionGroupItemResponse(
        item=food_item_read_from_orm(item),
        substitution_group=group,
        related_items=[food_item_read_from_orm(i) for i in related_items],
    )


@router.get(
    "/categories",
    summary="Leaf product categories",
)
async def get_categories(
    db: AsyncSession = Depends(get_db),
):
    """Subcategories only (rows with ``parent_id`` set). Top-level aisles like *Kolonial*
    are structural parents and are not valid ``food_items.main_category_id`` targets, so they
    are omitted here. Use ``GET /catalogue/snapshot`` for the full taxonomy tree.
    """
    stmt = (
        select(Category)
        .where(Category.parent_id.is_not(None))
        .order_by(Category.id)
    )
    result = await db.execute(stmt)
    return list(result.scalars().unique().all())

@router.get(
    "/categories/{category_id}",
    response_model=CategoryResponse,
    summary="Items filtered by category",
)
async def get_category(
    category_id: int,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    # Ensure the category exists
    category_obj = await db.get(Category, category_id)
    if category_obj is None:
        raise HTTPException(status_code=404, detail="Category not found")

    stmt = (
        select(FoodItem)
        .where(FoodItem.main_category_id == category_id)
        .options(
            selectinload(FoodItem.category),
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            ),
        )
        .order_by(FoodItem.created_at)
    )

    result = await db.execute(stmt)
    items = list(result.scalars().unique().all())

    return CategoryResponse(
        id=category_id,
        version=_catalogue_version(items),
        category=category_obj.name,
        item_count=len(items),
        items=[food_item_read_from_orm(item) for item in items],
    )


@router.get(
    "/item/{item_id}",
    response_model=FoodItemRead,
    summary="Single item by UUID",
)
async def get_item(
    item_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FoodItemRead:
    stmt = (
        select(FoodItem)
        .where(FoodItem.id == item_id)
        .options(
            selectinload(FoodItem.category),
            selectinload(FoodItem.substitution_groups).selectinload(
                FoodItemSubstitutionGroup.substitution_group
            ),
        )
    )

    result = await db.execute(stmt)
    item = result.scalar_one_or_none()

    if item is None:
        raise HTTPException(status_code=404, detail="Food item not found")

    return food_item_read_from_orm(item)