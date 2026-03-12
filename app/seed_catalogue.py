from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.database import AsyncSessionLocal
from app.api.schemas.category import Category
from app.api.schemas.food_item import FoodItem
from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup
from app.api.schemas.substitution_group import SubstitutionGroup


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "synthetic_food_catalogue_100_items_v2.json"

def _codeify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _titleize_group(code: str) -> str:
    return code.replace("_", " ").title()


def _per_serving(value_per_100g: float | None, serving_size_g: float) -> float | None:
    if value_per_100g is None:
        return None
    return round(float(value_per_100g) * float(serving_size_g) / 100.0, 2)


async def seed_catalogue() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}")

    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = raw.get("items", [])
    groups_payload = raw.get("substitution_group_taxonomy", {}).get("groups", [])
    if not items:
        print("No items found in dataset; nothing to seed.")
        return

    async with AsyncSessionLocal() as session:
        existing_count = await session.scalar(select(func.count()).select_from(FoodItem))
        if existing_count and existing_count > 0:
            print(f"Catalogue already seeded with {existing_count} food items; skipping.")
            return

        category_by_name: dict[str, Category] = {}
        for category_name in sorted({item["category"] for item in items}):
            category = Category(code=_codeify(category_name), name=category_name)
            session.add(category)
            await session.flush()
            category_by_name[category_name] = category

        group_by_code: dict[str, SubstitutionGroup] = {}
        for group in groups_payload:
            code = group["id"]
            db_group = SubstitutionGroup(
                code=code,
                name=_titleize_group(code),
                description=(
                    f"Synthetic substitution group for {group['category']} "
                    f"(expected {group.get('count', 0)} items)."
                ),
            )
            session.add(db_group)
            await session.flush()
            group_by_code[code] = db_group

        for row in items:
            serving_size_g = float(row["serving_size_g"])
            category = category_by_name[row["category"]]
            substitution_group = group_by_code[row["substitution_group"]]

            is_vegan = bool(row.get("is_vegan", False))
            is_vegetarian = bool(row.get("is_vegetarian", False))
            category_name = row["category"].lower()

            food_item = FoodItem(
                external_code=row["id"],
                name=row["name"],
                category_id=category.id,
                market=row.get("market"),
                brand=row.get("brand"),
                price_eur=float(row["price_eur"]),
                serving_size_g=serving_size_g,
                co2_kg_per_kg=float(row["co2_kg_per_kg"]),
                co2_kg_per_serving=float(row["co2_kg_per_serving"]),
                calories_kcal=_per_serving(row.get("calories_per_100g"), serving_size_g),
                protein_g=_per_serving(row.get("protein_g_per_100g"), serving_size_g),
                fat_g=_per_serving(row.get("fat_g_per_100g"), serving_size_g),
                carbs_g=_per_serving(row.get("carbs_g_per_100g"), serving_size_g),
                fiber_g=_per_serving(row.get("fiber_g_per_100g"), serving_size_g),
                sugar_g=_per_serving(row.get("sugar_g_per_100g"), serving_size_g),
                processing_level=row.get("processing_level"),
                is_meat="meat" in category_name or "poultry" in category_name,
                is_dairy="dairy" in category_name,
                is_plant_based=is_vegan,
                is_vegan=is_vegan,
                is_vegetarian=is_vegetarian,
                is_gluten_free=bool(row.get("is_gluten_free", False)),
                allergens=row.get("allergens", []),
                #item_metadata={
                #    "source": "synthetic_food_catalogue_100_items_v2",
                #    "raw_item": row,
                #    "dataset_metadata": raw.get("metadata", {}),
                #    "feature_flags": raw.get("features", {}),
                #},
            )
            session.add(food_item)
            await session.flush()

            session.add(
                FoodItemSubstitutionGroup(
                    food_item_id=food_item.id,
                    substitution_group_id=substitution_group.id,
                    group_priority=1,
                )
            )

        await session.commit()
        print(f"Seeded {len(items)} food items across {len(category_by_name)} categories.")
