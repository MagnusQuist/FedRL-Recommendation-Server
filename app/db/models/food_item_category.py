from __future__ import annotations

from typing import TYPE_CHECKING

from app.db import Base

from sqlalchemy import (
    String,
    ForeignKey,
    Integer,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.db.models.category import Category
    from app.db.models.food_item import FoodItem


class FoodItemCategory(Base):
    __tablename__ = "food_item_categories"

    product_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("food_items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("categories.category_id", ondelete="CASCADE"),
        primary_key=True,
    )

    food_item: Mapped["FoodItem"] = relationship(back_populates="food_item_categories")
    category: Mapped["Category"] = relationship(back_populates="food_item_categories")

    __table_args__ = (
        Index("ix_food_item_categories_product_id", "product_id"),
        Index("ix_food_item_categories_category_id", "category_id"),
    )