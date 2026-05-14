import gzip
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.snapshot import DatabaseSnapshotResponse
from app.services.snapshot import build_db_snapshot

router = APIRouter(prefix="/dev/db")


@router.get("/snapshot/json", response_model=DatabaseSnapshotResponse)
async def db_snapshot_json(
    max_rows_per_table: int | None = Query(default=None, ge=1, le=10_000),
    include_model_blobs: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
) -> DatabaseSnapshotResponse:
    """JSON snapshot of key database tables. Development endpoint."""
    return await build_db_snapshot(
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
    """Same data as /snapshot/json, returned as a downloadable attachment."""
    snapshot = await build_db_snapshot(
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
