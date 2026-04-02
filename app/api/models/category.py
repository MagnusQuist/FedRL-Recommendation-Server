from pydantic import BaseModel, ConfigDict, Field


class Category(BaseModel):
    id: int
    code: str = Field(..., description="Short category code, e.g. 'dairy_products'.")
    name: str = Field(..., description="Human-readable category name, e.g. 'Dairy products'.")
    parent_id: int | None = Field(
        default=None,
        description="Parent category id for subcategories; null for top-level aisles.",
    )

    model_config = ConfigDict(from_attributes=True)

