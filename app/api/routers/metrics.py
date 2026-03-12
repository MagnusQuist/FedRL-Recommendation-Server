import asyncio
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_db


router = APIRouter(prefix="/dev")


def get_aggregator(request: Request):
    return request.app.state.aggregator


async def _collect_metrics(aggregator) -> dict:
    """
    Build the metrics payload used by both the JSON and WebSocket endpoints.
    """
    async with AsyncSessionLocal() as db:
        latest = await aggregator.get_current_version(db)  # type: ignore[arg-type]

    if latest is None:
        backbone_info = None
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

    agg_state = aggregator.metrics_snapshot()

    return {
        "backbone": backbone_info,
        "aggregator": agg_state,
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
    # WebSocket connections do not have a standard Request object attached,
    # but FastAPI exposes the ASGI app via websocket.app.
    aggregator = websocket.app.state.aggregator  # type: ignore[attr-defined]

    try:
        while True:
            payload = await _collect_metrics(aggregator)
            await websocket.send_json(payload)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        # Client disconnected cleanly; nothing to log for dev dashboard.
        return


METRICS_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "metrics.html"


@router.get("/metrics")
async def metrics_page():
    """
    Simple development dashboard that connects to /dev/metrics/ws via WebSocket
    and renders key FL metrics in a minimal UI.
    """
    return FileResponse(METRICS_HTML_PATH)

