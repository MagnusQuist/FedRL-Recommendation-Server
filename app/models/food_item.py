import uuid
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class FoodItemBase(BaseModel):
    name: str = Field(..., description="Display name, e.g. 'Oat Milk 1L'.")
    category: str = Field(..., description="Food category, e.g. 'dairy_alternatives'.")
    co2e_score: float = Field(..., ge=0.0, le=1.0, description="Normalised score 0–1. Higher = more sustainable.")
    co2e_kg_per_kg: float = Field(..., ge=0.0, description="Raw CO2e in kg per kg (Poore & Nemecek 2018).")
    price: float = Field(..., ge=0.0, description="Price in DKK.")
    unit: str = Field(..., description="Unit of sale, e.g. '1L', '500g'.")
    alternative_ids: list[uuid.UUID] = Field(default_factory=list, description="IDs of greener alternatives in same category.")


class FoodItemCreate(FoodItemBase):
    pass


class FoodItemRead(FoodItemBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
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