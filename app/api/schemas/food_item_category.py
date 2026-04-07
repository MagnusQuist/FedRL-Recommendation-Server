from pydantic import BaseModel

from .common import ORMModel


class FoodItemCategoryBase(BaseModel):
    product_id: int
    category_id: int


class FoodItemCategoryCreate(FoodItemCategoryBase):
    pass


class FoodItemCategoryRead(ORMModel):
    product_id: int
    category_id: int