from __future__ import annotations

from app.db import Base


from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Integer,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship


class FoodItemCategory(Base):
    __tablename__ = "food_item_categories"

    product_id: Mapped[int] = mapped_column(
        BigInteger,
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