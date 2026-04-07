from __future__ import annotations

from app.db import Base

from sqlalchemy import (
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

class SubstitutionGroup(Base):
    __tablename__ = "substitution_groups"

    substitution_group_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)

    substitution_group_items: Mapped[list[SubstitutionGroupItem]] = relationship(
        back_populates="substitution_group",
        cascade="all, delete-orphan",
    )

    food_items: Mapped[list["FoodItem"]] = relationship(
        secondary="substitution_group_items",
        back_populates="substitution_groups",
        viewonly=True,
    )