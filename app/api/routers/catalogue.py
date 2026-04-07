from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db


router = APIRouter(prefix="/catalogue")

@router.get("/snapshot")
async def get_catalogue_snapshot(db: AsyncSession = Depends(get_db)):
    ...
    # """Retrieve the catalogue snapshot."""

    # # Items
    # stmt_items = (
    #     select(FoodItem)
    #     .options(
    #         selectinload(FoodItem.category),
    #         selectinload(FoodItem.substitution_groups).selectinload(
    #             FoodItemSubstitutionGroup.substitution_group
    #         ),
    #     )
    #     .order_by(FoodItem.created_at)
    # )
    # result_items = await db.execute(stmt_items)
    # items: list[FoodItem] = result_items.scalars().unique().all()
    # payload: list[FoodItemRead] = [food_item_read_from_orm(item) for item in items]

    # # Categories and substitution groups
    # categories_result = await db.execute(select(Category))
    # categories: list[Category] = categories_result.scalars().unique().all()

    # substitution_groups_stmt = (
    #     select(SubstitutionGroup)
    #     .options(
    #         selectinload(SubstitutionGroup.food_items)
    #         .selectinload(FoodItemSubstitutionGroup.food_item)
    #     )
    #     .order_by(SubstitutionGroup.id)
    # )

    # substitution_groups_result = await db.execute(substitution_groups_stmt)
    # substitution_groups: list[SubstitutionGroup] = substitution_groups_result.scalars().unique().all()

    # substitution_groups_payload: list[SubstitutionGroups] = [
    #     SubstitutionGroups(id=group.id, name=group.name, item_ids=[rel.food_item.id for rel in group.food_items])
    #     for group in substitution_groups
    # ]

    # return CatalogueSnapshot(
    #     version=_catalogue_version(items),
    #     categories=categories,
    #     substitution_groups=substitution_groups_payload,
    #     items=payload,
    # )