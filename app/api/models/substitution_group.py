from pydantic import BaseModel, Field


class SubstitutionGroups(BaseModel):
    id: int
    name: str = Field(..., description="Group name, usually same as code.")
    item_ids: list[int] = Field(
        ...,
        description="Foreign key to food_items.id.",
    )

