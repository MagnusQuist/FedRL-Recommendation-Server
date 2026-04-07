from __future__ import annotations

from uuid import uuid4

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.util import defaultdict

from app.db.models.catalogue_version import CatalogueVersion
from app.db.models.category import Category
from app.db.models.food_item import FoodItem
from app.db.models.food_item_category import FoodItemCategory
from app.db.models.substitution_group_item import SubstitutionGroupItem
from app.api.schemas.catalogue_snapshot import CatalogueSnapshotResponse
from app.api.schemas.catalogue_version import CatalogueVersionResponse
from app.api.schemas.category import CategoryRead
from app.api.schemas.food_item import FoodItemRead


async def get_or_create_catalogue_version(db: AsyncSession) -> CatalogueVersion:
    result = await db.execute(
        select(CatalogueVersion).order_by(desc(CatalogueVersion.generated_at)).limit(1)
    )
    current = result.scalar_one_or_none()

    if current is not None:
        return current

    current = CatalogueVersion(version=str(uuid4()))
    db.add(current)
    await db.commit()
    await db.refresh(current)
    return current


async def bump_catalogue_version(db: AsyncSession) -> CatalogueVersion:
    new_version = CatalogueVersion(version=str(uuid4()))
    db.add(new_version)
    await db.commit()
    await db.refresh(new_version)
    return new_version


async def get_version(db: AsyncSession) -> CatalogueVersionResponse:
    current = await get_or_create_catalogue_version(db)
    return CatalogueVersionResponse.model_validate(current)

async def build_catalogue_snapshot(db: AsyncSession) -> CatalogueSnapshotResponse:
    current_version = await get_or_create_catalogue_version(db)

    food_items_result = await db.execute(select(FoodItem).order_by(FoodItem.id))
    categories_result = await db.execute(select(Category).order_by(Category.category_id))

    food_item_categories_result = await db.execute(select(FoodItemCategory))
    substitution_group_items_result = await db.execute(select(SubstitutionGroupItem))

    food_items = food_items_result.scalars().all()
    categories = categories_result.scalars().all()
    food_item_categories_rows = food_item_categories_result.scalars().all()
    substitution_group_items_rows = substitution_group_items_result.scalars().all()

    food_item_categories: dict[int, list[int]] = defaultdict(list)
    for row in food_item_categories_rows:
        food_item_categories[row.category_id].append(row.product_id)

    substitution_group_items: dict[int, list[int]] = defaultdict(list)
    for row in substitution_group_items_rows:
        substitution_group_items[row.substitution_group_id].append(row.product_id)

    return CatalogueSnapshotResponse(
        version=current_version.version,
        generated_at=current_version.generated_at,
        food_items=[FoodItemRead.model_validate(item) for item in food_items],
        categories=[CategoryRead.model_validate(category) for category in categories],
        food_item_categories=dict(food_item_categories),
        substitution_group_items=dict(substitution_group_items),
    )