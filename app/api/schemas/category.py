from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ORMModel


class CategoryBase(BaseModel):
    name: str = Field(..., max_length=120)
    slug: str = Field(..., max_length=140)


class CategoryCreate(CategoryBase):
    pass


class CategoryUpdate(BaseModel):
    name: str | None = Field(None, max_length=120)
    slug: str | None = Field(None, max_length=140)


class CategoryRead(ORMModel):
    category_id: int
    name: str
    slug: str


class CategorySummary(ORMModel):
    category_id: int
    name: str
    slug: str


class CategoryDetail(CategoryRead):
    food_items: list["FoodItemSummary"] = Field(default_factory=list)


from .foodItem import FoodItemSummary

CategoryDetail.model_rebuild()