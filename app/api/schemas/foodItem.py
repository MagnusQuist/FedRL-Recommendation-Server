from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from .common import ORMModel


class FoodItemBase(BaseModel):
    name: str = Field(..., max_length=255)
    brand: str | None = Field(None, max_length=255)

    product_weight_in_g: int | None = None

    co2_kg_per_kg: Decimal | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: Decimal | None = None
    fat_g_per_100g: Decimal | None = None
    carbs_g_per_100g: Decimal | None = None
    fiber_g_per_100g: Decimal | None = None
    salt_g_per_100g: Decimal | None = None

    is_liquid: bool = False
    is_gluten_free: bool = False
    is_sugar_free: bool = False
    is_oekomærket_eu: bool = False
    is_oekomærket_dk: bool = False
    is_noeglehulsmaerket: bool = False
    is_fuldkornsmaerket: bool = False
    is_frozen: bool = False
    is_msc_maerket: bool = False
    is_fairtrade: bool = False
    is_rainforest_alliance: bool = False
    is_danish: bool = False

    price_dkk: Decimal | None = None


class FoodItemCreate(FoodItemBase):
    id: int


class FoodItemUpdate(BaseModel):
    name: str | None = Field(None, max_length=255)
    brand: str | None = Field(None, max_length=255)

    product_weight_in_g: int | None = None

    co2_kg_per_kg: Decimal | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: Decimal | None = None
    fat_g_per_100g: Decimal | None = None
    carbs_g_per_100g: Decimal | None = None
    fiber_g_per_100g: Decimal | None = None
    salt_g_per_100g: Decimal | None = None

    is_liquid: bool | None = None
    is_gluten_free: bool | None = None
    is_sugar_free: bool | None = None
    is_oekomærket_eu: bool | None = None
    is_oekomærket_dk: bool | None = None
    is_noeglehulsmaerket: bool | None = None
    is_fuldkornsmaerket: bool | None = None
    is_frozen: bool | None = None
    is_msc_maerket: bool | None = None
    is_fairtrade: bool | None = None
    is_rainforest_alliance: bool | None = None
    is_danish: bool | None = None

    price_dkk: Decimal | None = None


class FoodItemRead(ORMModel):
    id: int
    name: str
    brand: str | None = None

    product_weight_in_g: int | None = None

    co2_kg_per_kg: Decimal | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: Decimal | None = None
    fat_g_per_100g: Decimal | None = None
    carbs_g_per_100g: Decimal | None = None
    fiber_g_per_100g: Decimal | None = None
    salt_g_per_100g: Decimal | None = None

    is_liquid: bool
    is_gluten_free: bool
    is_sugar_free: bool
    is_oekomærket_eu: bool
    is_oekomærket_dk: bool
    is_noeglehulsmaerket: bool
    is_fuldkornsmaerket: bool
    is_frozen: bool
    is_msc_maerket: bool
    is_fairtrade: bool
    is_rainforest_alliance: bool
    is_danish: bool

    price_dkk: Decimal | None = None


class FoodItemSummary(ORMModel):
    id: int
    name: str
    brand: str | None = None
    price_dkk: Decimal | None = None
    co2_kg_per_kg: Decimal | None = None


class FoodItemDetail(FoodItemRead):
    categories: list["CategorySummary"] = Field(default_factory=list)
    substitution_groups: list["SubstitutionGroupSummary"] = Field(default_factory=list)


from .category import CategorySummary
from .substitutionGroup import SubstitutionGroupSummary

FoodItemDetail.model_rebuild()