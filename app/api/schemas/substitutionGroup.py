from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ORMModel


class SubstitutionGroupBase(BaseModel):
    name: str = Field(..., max_length=160)


class SubstitutionGroupCreate(SubstitutionGroupBase):
    pass


class SubstitutionGroupUpdate(BaseModel):
    name: str | None = Field(None, max_length=160)


class SubstitutionGroupRead(ORMModel):
    substitution_group_id: int
    name: str


class SubstitutionGroupSummary(ORMModel):
    substitution_group_id: int
    name: str


class SubstitutionGroupDetail(SubstitutionGroupRead):
    food_items: list["FoodItemSummary"] = Field(default_factory=list)


from .foodItem import FoodItemSummary

SubstitutionGroupDetail.model_rebuild()