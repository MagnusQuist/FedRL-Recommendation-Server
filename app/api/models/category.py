from pydantic import BaseModel, Field

class Subcategory(BaseModel):
    id: int
    name: str = Field(..., description="Subcategory name, e.g. 'Milk'.")

class Category(BaseModel):
    id: int
    name: str = Field(..., description="Category name, e.g. 'Dairy products'.")
    subcategories: list[Subcategory] = Field(
        default_factory=list,
        description="Subcategories of the category.",
    )

