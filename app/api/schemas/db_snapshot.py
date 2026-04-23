from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
