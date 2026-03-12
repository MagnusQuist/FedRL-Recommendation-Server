import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.fl.aggregator import MIN_CLIENTS_PER_ROUND, ROUND_TIMEOUT_SECONDS
from app.api.models.backbone import BackboneDownload, BackboneUpload, RoundStatus, UploadAck

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backbone")


def get_aggregator(request: Request):
    """FastAPI dependency — retrieves the FLAggregator singleton from app state."""
    return request.app.state.aggregator


@router.get(
    "/status",
    response_model=RoundStatus,
    summary="Aggregation queue status",
)
async def backbone_status(
    algorithm: str = "ts",
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """Returns the current state of the FL aggregation queue."""
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


@router.get(
    "/version",
    summary="Global backbone version",
)
async def backbone_version(
    db: AsyncSession = Depends(get_db),
    algorithm: str = "ts",
    aggregator=Depends(get_aggregator),
):
    """Return the current global backbone version (if any)."""
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

    return {
        "version": latest.version,
        "algorithm": latest.algorithm,
        "client_count": latest.client_count,
        "total_interactions": latest.total_interactions,
    }


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
    """Return the current global backbone if a newer version exists."""
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

    return BackboneDownload(
        version=latest.version,
        algorithm=latest.algorithm,
        client_count=latest.client_count,
        total_interactions=latest.total_interactions,
        backbone_weights=latest.weights_blob,
    )


@router.post(
    "/model",
    response_model=UploadAck,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload backbone weights for FedAvg aggregation",
)
async def upload_backbone(
    payload: BackboneUpload,
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """Accept a backbone weight upload from a Raspberry Pi client."""
    logger.info(
        "Received upload from client_id='%s' backbone_version=%d n_k=%d algorithm=%s",
        payload.client_id,
        payload.backbone_version,
        payload.interaction_count,
        payload.algorithm,
    )

    # The payload includes a base64-encoded, gzip-compressed JSON blob of the
    # backbone weight tensors (matching GET /backbone/model). Decode it here
    # before passing on to the aggregator.
    weights_dict = payload.backbone_weights
    if isinstance(weights_dict, str):
        from app.fl.aggregator import decode_backbone_blob

        weights_dict = decode_backbone_blob(weights_dict)

    round_triggered, queued = await aggregator.enqueue(
        client_id=payload.client_id,
        backbone_version=payload.backbone_version,
        interaction_count=payload.interaction_count,
        algorithm=payload.algorithm,
        weights_dict=weights_dict,
        db=db,
    )

    return UploadAck(
        status="queued",
        client_id=payload.client_id,
        queued_clients=queued,
        round_triggered=round_triggered,
    )
