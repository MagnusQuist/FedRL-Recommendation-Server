"""Response models for substitution groups."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from app.api.models.substitution_group import SubstitutionGroup


class SubstitutionGroupWithItems(BaseModel):
    """Substitution group payload including the UUIDs of linked food items."""

    group: SubstitutionGroup
    items: list[UUID]
