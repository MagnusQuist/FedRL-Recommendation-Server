import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.fl.aggregator import MIN_CLIENTS_PER_ROUND, ROUND_TIMEOUT_SECONDS
from app.models.backbone import BackboneDownload, BackboneUpload, RoundStatus, UploadAck

logger = logging.getLogger(__name__)
router = APIRouter()


def get_aggregator(request: Request):
    """FastAPI dependency — retrieves the FLAggregator singleton from app state."""
    return request.app.state.aggregator


# ---------------------------------------------------------------------------
# POST /fl/upload
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=UploadAck,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload backbone weights for FedAvg aggregation",
)
async def upload_backbone(
    payload: BackboneUpload,
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """
    Accepts a backbone weight upload from a Raspberry Pi client.

    The payload is validated against the known backbone parameter keys before
    being queued. Any upload containing unexpected keys (e.g. local head params)
    is rejected with 422 Unprocessable Entity — this is the server-side privacy
    enforcement described in the system design.

    If enqueueing this upload triggers a FedAvg round (queue >= min_clients_per_round),
    aggregation runs synchronously before responding. The client is informed via
    the round_triggered field in the response.
    """
    logger.info(
        "Received upload from client_id='%s' backbone_version=%d n_k=%d algorithm=%s",
        payload.client_id,
        payload.backbone_version,
        payload.interaction_count,
        payload.algorithm,
    )

    round_triggered, queued = await aggregator.enqueue(
        client_id=payload.client_id,
        backbone_version=payload.backbone_version,
        interaction_count=payload.interaction_count,
        algorithm=payload.algorithm,
        weights_dict=payload.backbone_weights,
        db=db,
    )

    return UploadAck(
        status="queued",
        client_id=payload.client_id,
        queued_clients=queued,
        round_triggered=round_triggered,
    )


# ---------------------------------------------------------------------------
# GET /fl/model
# ---------------------------------------------------------------------------
@router.get(
    "/model",
    response_model=BackboneDownload,
    summary="Download current global backbone",
    responses={304: {"description": "Client already has the latest version."}},
)
async def download_backbone(
    since: int = 0,
    algorithm: str = "ts",
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """
    Returns the current global backbone if a newer version exists.

    Query params:
        since     -- client's current backbone version. Returns 304 if up to date.
        algorithm -- 'ts' or 'dqn'.

    On app launch, clients call this with since=0 to always receive a backbone.
    """
    latest = await aggregator.get_current_version(db)

    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No global backbone exists yet. Run the seed script first.",
        )

    if latest.algorithm != algorithm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No backbone found for algorithm '{algorithm}'.",
        )

    if latest.version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    # Expose the stored gzip-compressed, base64-encoded weights blob directly.
    # Clients can decode base64 and then gunzip to recover the original tensors.
    return BackboneDownload(
        version=latest.version,
        algorithm=latest.algorithm,
        client_count=latest.client_count,
        total_interactions=latest.total_interactions,
        backbone_weights=latest.weights_blob,
    )


# ---------------------------------------------------------------------------
# GET /fl/status
# ---------------------------------------------------------------------------
@router.get(
    "/status",
    response_model=RoundStatus,
    summary="Aggregation queue status",
)
async def fl_status(
    algorithm: str = "ts",
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """
    Returns the current state of the FL aggregation queue.
    Use this during experiments to monitor uploads and completed rounds.
    """
    latest = await aggregator.get_current_version(db)
    current_version = latest.version if latest else 0

    return RoundStatus(
        current_version=current_version,
        algorithm=algorithm,
        queued_clients=aggregator.queued_client_ids(),
        total_rounds_completed=aggregator.rounds_completed,
        min_clients_per_round=MIN_CLIENTS_PER_ROUND,
        round_timeout_seconds=ROUND_TIMEOUT_SECONDS,
    )