from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CatalogueSnapshotResponse(BaseModel):
    version: str
    generated_at: datetime

    food_items: list["FoodItemRead"] = Field(default_factory=list)
    categories: list["CategoryRead"] = Field(default_factory=list)

    food_item_categories: dict[int, list[int]] = Field(default_factory=dict)
    substitution_group_items: dict[int, list[int]] = Field(default_factory=dict)

from .category import CategoryRead # noqa: E402
from .food_item import FoodItemRead # noqa: E402

CatalogueSnapshotResponse.model_rebuild()