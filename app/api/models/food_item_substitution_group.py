from uuid import UUID
from pydantic import BaseModel, ConfigDict


class FoodItemSubstitutionGroupBase(BaseModel):
    substitution_group_id: int
    group_priority: int = 1


class FoodItemSubstitutionGroupCreate(FoodItemSubstitutionGroupBase):
    food_item_id: UUID


class FoodItemSubstitutionGroupRead(FoodItemSubstitutionGroupBase):
    food_item_id: UUID

    model_config = ConfigDict(from_attributes=True)