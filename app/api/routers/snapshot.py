from __future__ import annotations

import gzip
import json
from datetime import date, datetime, time, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.db_snapshot import DatabaseSnapshotResponse, DatabaseTableSnapshot
from app.db import get_db
from app.db.models.aggregation_events import AggregationEvent
from app.db.models.catalogue_version import CatalogueVersion
from app.db.models.category import Category
from app.db.models.centralized import CentralizedModel
from app.db.models.centralized_training_events import CentralizedTrainingEvent
from app.db.models.federated import FederatedModel
from app.db.models.food_item import FoodItem
from app.db.models.food_item_category import FoodItemCategory
from app.db.models.substitution_group import SubstitutionGroup
from app.db.models.substitution_group_item import SubstitutionGroupItem

router = APIRouter(prefix="/dev/db")

_TABLE_MODELS: list[tuple[str, type[Any]]] = [
    ("catalogue_versions", CatalogueVersion),
    ("categories", Category),
    ("food_items", FoodItem),
    ("food_item_categories", FoodItemCategory),
    ("substitution_groups", SubstitutionGroup),
    ("substitution_group_items", SubstitutionGroupItem),
    ("federated_model_versions", FederatedModel),
    ("centralized_model_versions", CentralizedModel),
    ("aggregation_events", AggregationEvent),
    ("centralized_training_events", CentralizedTrainingEvent),
]

_BLOB_COLUMNS = {
    "weights_blob",
    "backbone_blob",
    "reward_predictor_blob",
    "item_head_blob",
    "price_head_blob",
    "nudge_head_blob",
    "tuple_pool_blob",
}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return value


def _extract_row(
    instance: Any,
    *,
    include_model_blobs: bool,
) -> tuple[dict[str, Any], set[str]]:
    mapper = inspect(instance.__class__)
    row: dict[str, Any] = {}
    omitted: set[str] = set()

    for attr in mapper.column_attrs:
        key = attr.key
        if not include_model_blobs and key in _BLOB_COLUMNS:
            omitted.add(key)
            continue
        row[key] = _serialize_value(getattr(instance, key))

    return row, omitted


async def _snapshot_table(
    db: AsyncSession,
    *,
    table_name: str,
    model: type[Any],
    max_rows_per_table: int | None,
    include_model_blobs: bool,
) -> DatabaseTableSnapshot:
    count_result = await db.execute(select(func.count()).select_from(model))
    row_count = int(count_result.scalar_one())

    mapper = inspect(model)
    stmt = select(model)
    for pk_column in mapper.primary_key:
        stmt = stmt.order_by(pk_column.asc())
    if max_rows_per_table is not None:
        stmt = stmt.limit(max_rows_per_table)

    rows_result = await db.execute(stmt)
    instances = rows_result.scalars().all()

    rows: list[dict[str, Any]] = []
    omitted_columns: set[str] = set()

    for instance in instances:
        row, omitted = _extract_row(instance, include_model_blobs=include_model_blobs)
        rows.append(row)
        omitted_columns.update(omitted)

    return DatabaseTableSnapshot(
        table=table_name,
        row_count=row_count,
        rows_included=len(rows),
        omitted_columns=sorted(omitted_columns),
        rows=rows,
    )


async def _build_snapshot(
    db: AsyncSession,
    *,
    max_rows_per_table: int | None,
    include_model_blobs: bool,
) -> DatabaseSnapshotResponse:
    tables: list[DatabaseTableSnapshot] = []
    for table_name, model in _TABLE_MODELS:
        table_snapshot = await _snapshot_table(
            db,
            table_name=table_name,
            model=model,
            max_rows_per_table=max_rows_per_table,
            include_model_blobs=include_model_blobs,
        )
        tables.append(table_snapshot)

    return DatabaseSnapshotResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        max_rows_per_table=max_rows_per_table,
        include_model_blobs=include_model_blobs,
        tables=tables,
    )


@router.get("/snapshot", response_model=DatabaseSnapshotResponse)
async def db_snapshot(
    max_rows_per_table: int | None = Query(default=None, ge=1, le=10_000),
    include_model_blobs: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
) -> DatabaseSnapshotResponse:
    """
    Development endpoint that returns a JSON snapshot of key database tables.

    By default, returns all rows from all selected tables, including model blobs.
    Use query params to limit row count or omit blob columns if needed.
    """
    return await _build_snapshot(
        db,
        max_rows_per_table=max_rows_per_table,
        include_model_blobs=include_model_blobs,
    )


@router.get("/snapshot/export")
async def export_db_snapshot(
    max_rows_per_table: int | None = Query(default=None, ge=1, le=10_000),
    include_model_blobs: bool = Query(default=True),
    compress: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    """
    Same data as /snapshot, but returned as a downloadable attachment.
    Set ``compress=true`` to return ``application/gzip``.
    """
    snapshot = await _build_snapshot(
        db,
        max_rows_per_table=max_rows_per_table,
        include_model_blobs=include_model_blobs,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = snapshot.model_dump(mode="json")

    if compress:
        filename = f"db_snapshot_{timestamp}.json.gz"
        compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        return Response(
            content=compressed,
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    filename = f"db_snapshot_{timestamp}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
