from __future__ import annotations

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator

from app.api.models.category import Category
from app.api.models.substitution_group import SubstitutionGroup
from app.api.models.substitution_group_with_items import SubstitutionGroupWithItems


class FoodItemBase(BaseModel):
    """Fields aligned with `data/product_items.json` (+ category_id FK for queries)."""

    external_code: str = Field(..., description="Product id string from dataset, e.g. '20807'.")
    name: str = Field(..., description="Display name.")

    category_id: int = Field(..., description="Foreign key to categories.id (leaf subcategory).")
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

    main_category: int = Field(..., description="Top-level category id from dataset.")
    sub_category: list[int] = Field(..., description="Subcategory id list from dataset.")

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

    substitution_group: int | None = Field(
        None,
        description="Primary substitution group id (from junction; same role as JSON field).",
    )

    @field_validator("sub_category", mode="before")
    @classmethod
    def coerce_sub_category(cls, v: object) -> list[int]:
        if v is None:
            return []
        return [int(x) for x in v]


class FoodItemRead(FoodItemBase):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: UUID
    created_at: datetime

    @computed_field
    @property
    def co2_kg_per_serving(self) -> float:
        """kg CO2e for the full package: co2_kg_per_kg × weight_kg."""
        return round(float(self.co2_kg_per_kg) * (float(self.product_weight_in_g) / 1000.0), 6)


class CatalogueResponse(BaseModel):
    """Wrapper returned by GET /catalogue — includes a version string so clients detect staleness."""

    version: str = Field(..., description="Catalogue version string (ISO timestamp of last seed/update).")
    item_count: int
    items: list[FoodItemRead]


class CategoryResponse(BaseModel):
    id: int
    version: str
    category: str
    item_count: int
    items: list[FoodItemRead]


class CatalogueSnapshot(BaseModel):
    """Combined snapshot of the catalogue (items + taxonomy)."""

    version: str
    categories: list["Category"]
    substitution_groups: list["SubstitutionGroupWithItems"]
    items: list[FoodItemRead]


class SubstitutionGroupItemResponse(BaseModel):
    """Response for `/substitution_groups/item`."""

    item: FoodItemRead
    substitution_group: "SubstitutionGroup"
    related_items: list[FoodItemRead]
