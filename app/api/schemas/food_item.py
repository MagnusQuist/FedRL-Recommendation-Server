from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ORMModel


class FoodItemBase(BaseModel):
    name: str = Field(..., max_length=255)
    brand: str | None = Field(None, max_length=255)

    product_weight_in_g: int | None = None

    co2e_kg_pr_item_kg: float | None = None
    estimated_co2e_kg_pr_item_weight_in_g: float | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: float | None = None
    fat_g_per_100g: float | None = None
    carbs_g_per_100g: float | None = None
    fiber_g_per_100g: float | None = None
    salt_g_per_100g: float | None = None

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

    price_dkk: float | None = None


class FoodItemCreate(FoodItemBase):
    id: str


class FoodItemUpdate(BaseModel):
    name: str | None = Field(None, max_length=255)
    brand: str | None = Field(None, max_length=255)

    product_weight_in_g: int | None = None

    co2e_kg_pr_item_kg: float | None = None
    estimated_co2e_kg_pr_item_weight_in_g: float | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: float | None = None
    fat_g_per_100g: float | None = None
    carbs_g_per_100g: float | None = None
    fiber_g_per_100g: float | None = None
    salt_g_per_100g: float | None = None

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

    price_dkk: float | None = None


class FoodItemRead(ORMModel):
    id: str
    name: str
    brand: str | None = None

    product_weight_in_g: int | None = None

    co2e_kg_pr_item_kg: float | None = None
    estimated_co2e_kg_pr_item_weight_in_g: float | None = None
    calories_per_100g: int | None = None
    protein_g_per_100g: float | None = None
    fat_g_per_100g: float | None = None
    carbs_g_per_100g: float | None = None
    fiber_g_per_100g: float | None = None
    salt_g_per_100g: float | None = None

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

    price_dkk: float | None = None


class FoodItemSummary(ORMModel):
    id: str
    name: str
    brand: str | None = None
    price_dkk: float | None = None
    estimated_co2e_kg_pr_item_weight_in_g: float | None = None


class FoodItemDetail(FoodItemRead):
    categories: list["CategorySummary"] = Field(default_factory=list)
    substitution_groups: list["SubstitutionGroupSummary"] = Field(default_factory=list)


from .category import CategorySummary  # noqa: E402
from .substitution_group import SubstitutionGroupSummary  # noqa: E402

FoodItemDetail.model_rebuild()