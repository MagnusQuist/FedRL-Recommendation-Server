from __future__ import annotations
from datetime import datetime

from .common import ORMModel


class CatalogueVersionResponse(ORMModel):
    version: str
    generated_at: datetime