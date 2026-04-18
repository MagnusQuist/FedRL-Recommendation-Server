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
    from app.db.models.food_item import FoodItem
    from app.db.models.substitution_group import SubstitutionGroup


class SubstitutionGroupItem(Base):
    __tablename__ = "substitution_group_items"

    substitution_group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("substitution_groups.substitution_group_id", ondelete="CASCADE"),
        primary_key=True,
    )
    product_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("food_items.id", ondelete="CASCADE"),
        primary_key=True,
    )

    substitution_group: Mapped["SubstitutionGroup"] = relationship(
        back_populates="substitution_group_items"
    )
    food_item: Mapped["FoodItem"] = relationship(back_populates="substitution_group_items")

    __table_args__ = (
        Index("ix_substitution_group_items_group_id", "substitution_group_id"),
        Index("ix_substitution_group_items_product_id", "product_id"),
    )