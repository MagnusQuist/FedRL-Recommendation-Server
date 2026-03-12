from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.api.schemas.food_item_substitution_group import FoodItemSubstitutionGroup


class SubstitutionGroup(Base):
    __tablename__ = "substitution_groups"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    food_items: Mapped[list["FoodItemSubstitutionGroup"]] = relationship(
        back_populates="substitution_group"
    )
