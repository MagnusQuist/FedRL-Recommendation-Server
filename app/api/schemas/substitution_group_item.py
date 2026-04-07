from pydantic import BaseModel

from .common import ORMModel


class SubstitutionGroupItemBase(BaseModel):
    substitution_group_id: int
    product_id: int


class SubstitutionGroupItemCreate(SubstitutionGroupItemBase):
    pass


class SubstitutionGroupItemRead(ORMModel):
    substitution_group_id: int
    product_id: int