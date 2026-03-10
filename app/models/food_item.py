import uuid
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class FoodItemBase(BaseModel):
    item_id: str = Field(..., description="Item ID, e.g. 'Ra00001-DK'.")
    name: str = Field(..., description="Display name, e.g. 'Oat Milk 1L'.")
    category: str = Field(..., description="Food category, e.g. 'dairy_alternatives'.")
    co2e_emission_tonnes: float = Field(..., ge=0.0, description="CO2e emission in tonnes.")
    price: float = Field(..., ge=0.0, description="Price in DKK.")
    alternative_ids: list[uuid.UUID] = Field(default_factory=list, description="IDs of greener alternatives in same category.")

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