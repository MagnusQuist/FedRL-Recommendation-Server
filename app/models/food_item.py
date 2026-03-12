from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class FoodItemBase(BaseModel):
    external_code: str = Field(..., description="External / dataset code, e.g. 'food_001'.")
    name: str = Field(..., description="Display name, e.g. 'Beef Burger Patty'.")

    category_id: int = Field(..., description="Foreign key to categories.id.")

    market: str | None = Field(None, description="Market / country code, e.g. 'DK'.")
    brand: str | None = Field(None, description="Optional brand name.")

    price_eur: float = Field(..., ge=0.0, description="Price of the item in EUR.")
    serving_size_g: float = Field(..., ge=0.0, description="Nominal serving size in grams.")

    co2_kg_per_kg: float = Field(..., ge=0.0, description="CO2 emission per kg of product (kg CO2e/kg).")
    co2_kg_per_serving: float = Field(..., ge=0.0, description="CO2 emission for a single serving (kg CO2e).")

    calories_kcal: float | None = Field(None, ge=0.0, description="Calories per serving in kcal.")
    protein_g: float | None = Field(None, ge=0.0, description="Protein grams per serving.")
    fat_g: float | None = Field(None, ge=0.0, description="Fat grams per serving.")
    carbs_g: float | None = Field(None, ge=0.0, description="Carbohydrate grams per serving.")
    fiber_g: float | None = Field(None, ge=0.0, description="Fibre grams per serving.")
    sugar_g: float | None = Field(None, ge=0.0, description="Sugar grams per serving.")

    is_meat: bool = Field(False)
    is_dairy: bool = Field(False)
    is_plant_based: bool = Field(False)
    is_vegan: bool = Field(False)
    is_vegetarian: bool = Field(False)
    is_gluten_free: bool = Field(False)

    allergens: list[str] = Field(default_factory=list, description="Array of allergen labels.")
    item_metadata: dict = Field(default_factory=dict, description="Arbitrary item metadata from the source catalogue.")


class FoodItemRead(FoodItemBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class CatalogueResponse(BaseModel):
    """Wrapper returned by GET /catalogue — includes a version string so clients detect staleness."""

    version: str = Field(..., description="Catalogue version string (ISO timestamp of last seed/update).")
    item_count: int
    items: list[FoodItemRead]


class CategoryResponse(BaseModel):
    version: str
    category: str
    item_count: int
    items: list[FoodItemRead]