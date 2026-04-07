from __future__ import annotations

from app.db import Base

from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    Integer,
    Numeric,
    String,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

class FoodItem(Base):
    __tablename__ = "food_items"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)

    product_weight_in_g: Mapped[int | None] = mapped_column(Integer, nullable=True)

    co2_kg_per_kg: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    calories_per_100g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protein_g_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    fat_g_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    carbs_g_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    fiber_g_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    salt_g_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    is_liquid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_gluten_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_sugar_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_oekomærket_eu: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_oekomærket_dk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_noeglehulsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_fuldkornsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_msc_maerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_fairtrade: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_rainforest_alliance: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_danish: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

    price_dkk: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    food_item_categories: Mapped[list[FoodItemCategory]] = relationship(
        back_populates="food_item",
        cascade="all, delete-orphan",
    )
    substitution_group_items: Mapped[list[SubstitutionGroupItem]] = relationship(
        back_populates="food_item",
        cascade="all, delete-orphan",
    )

    categories: Mapped[list[Category]] = relationship(
        secondary="food_item_categories",
        back_populates="food_items",
        viewonly=True,
    )
    substitution_groups: Mapped[list[SubstitutionGroup]] = relationship(
        secondary="substitution_group_items",
        back_populates="food_items",
        viewonly=True,
    )

    __table_args__ = (
        Index("ix_food_items_name", "name"),
        Index("ix_food_items_brand", "brand"),
    )