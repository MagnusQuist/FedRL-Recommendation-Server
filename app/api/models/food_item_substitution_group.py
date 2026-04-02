from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FoodItemSubstitutionGroupBase(BaseModel):
    """Aligned with ORM ``FoodItemSubstitutionGroup`` (composite PK on item + group)."""

    substitution_group_id: int = Field(
        ...,
        description="FK to substitution_groups.id.",
    )
    group_priority: int = Field(
        1,
        ge=1,
        description="Lower value = higher priority when choosing a primary group for an item.",
    )


class FoodItemSubstitutionGroupCreate(FoodItemSubstitutionGroupBase):
    food_item_id: UUID = Field(..., description="FK to food_items.id.")


class FoodItemSubstitutionGroupRead(FoodItemSubstitutionGroupBase):
    food_item_id: UUID = Field(..., description="FK to food_items.id.")

    model_config = ConfigDict(from_attributes=True)