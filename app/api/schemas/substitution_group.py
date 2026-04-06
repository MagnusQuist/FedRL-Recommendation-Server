from __future__ import annotations

from sqlalchemy import SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

class SubstitutionGroup(Base):
    __tablename__ = "substitution_groups"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True, comment="Group id.")
    name: Mapped[str] = mapped_column(String, nullable=False, comment="Group name.")
    item_ids: Mapped[list[int]] = mapped_column(SmallInteger, nullable=False, default=list, comment="Food item external code list.")

    def __repr__(self) -> str:
        return f"<SubstitutionGroup id={self.id} name={self.name!r} item_ids={self.item_ids!r}>"