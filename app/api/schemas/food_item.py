from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

if TYPE_CHECKING:
    from app.api.schemas.category import Category
    from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup


class FoodItem(Base):
    """ORM row aligned with `data/product_items.json` (plus UUID id and category FK)."""

    __tablename__ = "food_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    external_code: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        comment="Retailer / dataset product id as string, e.g. '20807'.",
    )
    name: Mapped[str] = mapped_column(String, nullable=False)

    category_id: Mapped[int] = mapped_column(
        SmallInteger,
        ForeignKey("categories.id", ondelete="RESTRICT"),
        nullable=False,
        comment="FK to leaf subcategory; mirrors first entry of sub_category JSON.",
    )
    category: Mapped["Category"] = relationship(back_populates="food_items")

    brand: Mapped[str | None] = mapped_column(String, nullable=True)

    price_dkk: Mapped[float] = mapped_column(
        Numeric(8, 2),
        nullable=False,
        comment="Shelf price in DKK.",
    )
    product_weight_in_g: Mapped[float] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        comment="Package / reference weight in grams (JSON string parsed to number).",
    )

    co2_kg_per_kg: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="CO2 emission per kg of product (kg CO2e/kg).",
    )

    calories_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    protein_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    fat_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    carbs_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    fiber_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    salt_g_per_100g: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)

    main_category: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sub_category: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of subcategory integer ids from the source JSON.",
    )

    is_liquid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_gluten_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_sugar_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_oekomærket_eu: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_oekomærket_dk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_noeglehulsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_fuldkornsmaerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_msc_maerket: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_fairtrade: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_rainforest_alliance: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_danish: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    substitution_groups: Mapped[list["FoodItemSubstitutionGroup"]] = relationship(
        back_populates="food_item",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<FoodItem id={self.id} external_code={self.external_code!r} "
            f"name={self.name!r} category_id={self.category_id}>"
        )
