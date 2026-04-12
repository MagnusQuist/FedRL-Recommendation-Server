import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.backbone import GlobalBackboneVersion
from app.db import AsyncSessionLocal, get_db
from app.db.seed_backbone import FEDERATED_ALGORITHM
from app.db.seed_status import is_database_seeded


router = APIRouter(prefix="/dev")


def get_aggregator(request: Request):
    return request.app.state.aggregator


async def _collect_metrics(aggregator) -> dict:
    """
    Build the metrics payload used by both the JSON and WebSocket endpoints.

    Metrics are reported per algorithm because backbone versioning and
    aggregation are now maintained independently for each algorithm.
    """
    async with AsyncSessionLocal() as db:
        backbone_by_algorithm: dict[str, dict | None] = {}

        for algorithm in (FEDERATED_ALGORITHM,):
            latest = await aggregator.get_current_version(db, algorithm=algorithm)  # type: ignore[arg-type]

            if latest is None:
                backbone_info = None
                last_update_age_seconds = None
            else:
                created_at = latest.created_at
                if created_at and created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

                backbone_info = {
                    "version": latest.version,
                    "algorithm": latest.algorithm,
                    "client_count": latest.client_count,
                    "total_interactions": latest.total_interactions,
                    "created_at": created_at.isoformat() if created_at else None,
                }

                if created_at:
                    last_update_age_seconds = (
                        datetime.now(timezone.utc) - created_at
                    ).total_seconds()
                else:
                    last_update_age_seconds = None

            result = await db.execute(
                select(func.count())
                .select_from(GlobalBackboneVersion)
                .where(GlobalBackboneVersion.algorithm == algorithm)
            )
            total_versions = int(result.scalar_one())

            backbone_by_algorithm[algorithm] = (
                {
                    **(backbone_info or {"algorithm": algorithm}),
                    "total_versions": total_versions,
                    "seconds_since_last_update": last_update_age_seconds,
                }
                if backbone_info is not None
                else {
                    "algorithm": algorithm,
                    "version": None,
                    "client_count": None,
                    "total_interactions": None,
                    "created_at": None,
                    "total_versions": total_versions,
                    "seconds_since_last_update": None,
                }
            )

    agg_state = aggregator.metrics_snapshot()
    seeded = await is_database_seeded()

    return {
        "backbones": backbone_by_algorithm,
        "aggregator": agg_state,
        "db_seeded": seeded,
    }


@router.get("/metrics/json")
async def metrics_json(
    db: AsyncSession = Depends(get_db),  # kept so the router remains compatible with FastAPI tooling
    aggregator=Depends(get_aggregator),
):
    """
    Development-only JSON view of key FL training metrics.
    This is intentionally lightweight and omits any model weights.
    """
    # We ignore the injected db here and instead use AsyncSessionLocal inside _collect_metrics
    # to avoid keeping a session open for the lifetime of a WebSocket connection.
    return await _collect_metrics(aggregator)


@router.websocket("/metrics/ws")
async def metrics_ws(websocket: WebSocket):
    """
    WebSocket endpoint that pushes live metrics snapshots to connected clients.
    The server still samples periodically, but over a single persistent socket.
    """
    await websocket.accept()
    aggregator = websocket.app.state.aggregator  # type: ignore[attr-defined]

    try:
        while True:
            payload = await _collect_metrics(aggregator)
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return


METRICS_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "metrics.html"


@router.get("/metrics")
async def metrics_page():
    """
    Simple development dashboard that connects to /dev/metrics/ws via WebSocket
    and renders key FL metrics in a minimal UI.
    """
    return FileResponse(METRICS_HTML_PATH)