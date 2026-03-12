from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

if TYPE_CHECKING:
    from app.api.schemas.food_item import FoodItem


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)

    food_items: Mapped[list["FoodItem"]] = relationship(back_populates="category")
