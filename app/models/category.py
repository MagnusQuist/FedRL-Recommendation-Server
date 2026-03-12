from pydantic import BaseModel, ConfigDict, Field


class Category(BaseModel):
    id: int
    code: str = Field(..., description="Short category code, e.g. 'dairy_products'.")
    name: str = Field(..., description="Human-readable category name, e.g. 'Dairy products'.")

    model_config = ConfigDict(from_attributes=True)

