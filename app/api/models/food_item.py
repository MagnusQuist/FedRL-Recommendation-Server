from __future__ import annotations

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from app.api.models.category import Category
from app.api.models.substitution_group import SubstitutionGroups


class FoodItemBase(BaseModel):
    """Fields aligned with `data/product_items.json` (source uses `sub_category`; we expose `sub_category_ids`)."""

    external_code: str = Field(..., description="Product id string from dataset, e.g. '20807'.")
    name: str = Field(..., description="Display name.")

    brand: str | None = Field(None, description="Optional brand name.")

    price_dkk: float = Field(..., ge=0.0, description="Price in DKK.")
    product_weight_in_g: float = Field(..., ge=0.0, description="Package weight in grams.")

    co2_kg_per_kg: float = Field(..., ge=0.0, description="kg CO2e per kg product.")

    calories_per_100g: float | None = Field(None, description="kcal per 100 g.")
    protein_g_per_100g: float | None = Field(None, description="Protein g per 100 g.")
    fat_g_per_100g: float | None = Field(None, description="Fat g per 100 g.")
    carbs_g_per_100g: float | None = Field(None, description="Carbohydrate g per 100 g.")
    fiber_g_per_100g: float | None = Field(None, description="Fibre g per 100 g.")
    salt_g_per_100g: float | None = Field(None, description="Salt g per 100 g.")

    main_category_id: int = Field(
        ...,
        description="Foreign key to categories.id (leaf subcategory); same as first entry of sub_category_ids.",
    )
    sub_category_ids: list[int] = Field(
        ...,
        description="Subcategory id list from dataset (JSON key `sub_category`); first id matches main_category_id.",
    )

    is_liquid: bool = Field(False)
    is_gluten_free: bool = Field(False)
    is_sugar_free: bool = Field(False)
    is_oekomærket_eu: bool = Field(False)
    is_oekomærket_dk: bool = Field(False)
    is_noeglehulsmaerket: bool = Field(False)
    is_fuldkornsmaerket: bool = Field(False)
    is_frozen: bool = Field(False)
    is_msc_maerket: bool = Field(False)
    is_fairtrade: bool = Field(False)
    is_rainforest_alliance: bool = Field(False)
    is_danish: bool = Field(False)

    substitution_group_ids: list[int] = Field(
        ...,
        description="Foreign key to substitution_groups.id.",
    )


class FoodItemRead(FoodItemBase):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: UUID
    created_at: datetime


class CatalogueSnapshot(BaseModel):
    """Combined snapshot of the catalogue (items + taxonomy)."""

    version: str
    categories: list["Category"]
    substitution_groups: list["SubstitutionGroups"]
    items: list[FoodItemRead]