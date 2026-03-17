from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
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
        comment="External code / synthetic ID from dataset, e.g. 'food_001'.",
    )
    name: Mapped[str] = mapped_column(String, nullable=False)

    category_id: Mapped[int] = mapped_column(
        SmallInteger,
        ForeignKey("categories.id", ondelete="RESTRICT"),
        nullable=False,
    )
    category: Mapped["Category"] = relationship(back_populates="food_items")

    market: Mapped[str | None] = mapped_column(String, nullable=True)
    brand: Mapped[str | None] = mapped_column(String, nullable=True)

    price_dkk: Mapped[float] = mapped_column(
        Numeric(8, 2),
        nullable=False,
        comment="Price of the item in EUR.",
    )
    serving_size_g: Mapped[float] = mapped_column(
        Numeric(8, 2),
        nullable=False,
        comment="Nominal serving size in grams.",
    )

    co2_kg_per_kg: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="CO2 emission per kg of product (kg CO2e/kg).",
    )
    co2_kg_per_serving: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        comment="Pre-computed CO2 emission for a single serving (kg CO2e).",
    )

    calories_kcal: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    protein_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    fat_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    carbs_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    fiber_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    sugar_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)

    processing_level: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_meat: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_dairy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_plant_based: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_vegan: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_vegetarian: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_gluten_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    allergens: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Array of allergen codes, e.g. ['milk', 'soy'].",
    )
    #item_metadata: Mapped[dict] = mapped_column(
    #    JSON,
    #    nullable=False,
    #    default=dict,
    #    comment="Arbitrary item metadata from the source catalogue.",
    #)

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
