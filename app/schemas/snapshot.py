"""DB snapshot response schemas and product label image enum."""

from __future__ import annotations

import re
import warnings
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── DB snapshot schemas ──────────────────────────────────────────────────────

class DatabaseTableSnapshot(BaseModel):
    table: str
    row_count: int
    rows_included: int
    omitted_columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class DatabaseSnapshotResponse(BaseModel):
    generated_at: str
    max_rows_per_table: int | None
    include_model_blobs: bool
    tables: list[DatabaseTableSnapshot]


# ── Product label image enum ─────────────────────────────────────────────────

# app/schemas/snapshot.py → parents[1] = app/
_PRODUCT_LABELS_DIR = Path(__file__).resolve().parents[1] / "static" / "product_labels"
PRODUCT_LABEL_EMPTY_VALUE = "__no_product_label_image__"


@lru_cache(maxsize=1)
def product_label_image_stems() -> tuple[str, ...]:
    if not _PRODUCT_LABELS_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in _PRODUCT_LABELS_DIR.glob("*.webp")))


def _member_name(stem: str) -> str:
    if stem.isidentifier() and not stem[0].isdigit():
        return stem.upper()
    s = re.sub(r"[^0-9a-zA-Z_]", "_", stem)
    if not s or not s.isidentifier() or s[0].isdigit():
        return f"ID_{s}"
    return s.upper()


def _build_enum() -> type[StrEnum]:
    used: set[str] = set()
    members: dict[str, str] = {}
    for stem in product_label_image_stems():
        base = _member_name(stem)
        name, n = base, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        members[name] = stem
    if not members:
        warnings.warn(f"No .webp files under {_PRODUCT_LABELS_DIR}; enum uses sentinel only.", stacklevel=2)
        members = {"_NO_ASSETS": PRODUCT_LABEL_EMPTY_VALUE}
    return StrEnum("ProductLabelImage", members)


ProductLabelImage = _build_enum()
