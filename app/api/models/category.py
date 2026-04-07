from __future__ import annotations

from app.db import Base


from sqlalchemy import (
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Category(Base):
    __tablename__ = "categories"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(140), nullable=False, unique=True)

    food_item_categories: Mapped[list[FoodItemCategory]] = relationship(
        back_populates="category",
        cascade="all, delete-orphan",
    )

    food_items: Mapped[list["FoodItem"]] = relationship(
        secondary="food_item_categories",
        back_populates="categories",
        viewonly=True,
    )