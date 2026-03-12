from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, SmallInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class FoodItemSubstitutionGroup(Base):
    __tablename__ = "food_item_substitution_groups"

    food_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("food_items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    substitution_group_id: Mapped[int] = mapped_column(
        SmallInteger,
        ForeignKey("substitution_groups.id", ondelete="CASCADE"),
        primary_key=True,
    )
    group_priority: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
    )

    food_item: Mapped["FoodItem"] = relationship(back_populates="substitution_groups")
    substitution_group: Mapped["SubstitutionGroup"] = relationship(
        back_populates="food_items"
    )
