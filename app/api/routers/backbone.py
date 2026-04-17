from app.logger import logger

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.backbones.aggregator import MIN_CLIENTS_PER_ROUND, ROUND_TIMEOUT_SECONDS
from app.api.schemas.backbone import BackboneDownload, BackboneUpload, RoundStatus, UploadAck
from app.db.seed_backbone import FEDERATED_ALGORITHM

router = APIRouter(prefix="/backbone")


def get_aggregator(request: Request):
    """FastAPI dependency — retrieves the FLAggregator singleton from app state."""
    return request.app.state.aggregator


def validate_algorithm_or_400(algorithm: str) -> str:
    if algorithm != FEDERATED_ALGORITHM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported algorithm '{algorithm}'. "
                f"Only the federated algorithm '{FEDERATED_ALGORITHM}' is supported."
            ),
        )
    return algorithm


@router.get(
    "/status",
    response_model=RoundStatus,
    summary="Aggregation queue status",
)
async def backbone_status(
    algorithm: str = Query("ts", description="Algorithm to inspect."),
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """Returns the current state of the FL aggregation queue for one algorithm."""
    algorithm = validate_algorithm_or_400(algorithm)

    latest = await aggregator.get_current_version(db, algorithm=algorithm)
    current_version = latest.version if latest else 0

    return RoundStatus(
        current_version=current_version,
        algorithm=algorithm,
        queued_clients=aggregator.queued_client_ids(algorithm=algorithm),
        total_rounds_completed=aggregator.rounds_completed(algorithm=algorithm),
        min_clients_per_round=MIN_CLIENTS_PER_ROUND,
        round_timeout_seconds=ROUND_TIMEOUT_SECONDS,
    )


@router.get(
    "/version",
    summary="Global backbone version",
)
async def backbone_version(
    db: AsyncSession = Depends(get_db),
    algorithm: str = Query("ts", description="Algorithm to inspect."),
    aggregator=Depends(get_aggregator),
):
    logger.info("Getting backbone verison")
    """Return the current global backbone version for the requested algorithm."""
    algorithm = validate_algorithm_or_400(algorithm)

    latest = await aggregator.get_current_version(db, algorithm=algorithm)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No backbone found for algorithm '{algorithm}'. Run the seed script first.",
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
    since: int = Query(0, ge=0, description="Client's current backbone version."),
    algorithm: str = Query("ts", description="Algorithm to download."),
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(get_aggregator),
):
    """Return the current global backbone for the requested algorithm if a newer version exists."""
    algorithm = validate_algorithm_or_400(algorithm)

    latest = await aggregator.get_current_version(db, algorithm=algorithm)

    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No backbone found for algorithm '{algorithm}'. Run the seed script first.",
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

    latest = await aggregator.get_current_version(db, algorithm=payload.algorithm)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No backbone found for algorithm '{payload.algorithm}'. Run the seed script first.",
        )

    # Optional strictness:
    # reject uploads against future versions or obviously invalid versions
    if payload.backbone_version < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="backbone_version must be >= 1.",
        )

    # The payload includes a base64-encoded, gzip-compressed JSON blob of the
    # backbone weight tensors (matching GET /backbone/model). Decode it here
    # before passing on to the aggregator.
    weights_dict = payload.backbone_weights
    if isinstance(weights_dict, str):
        from app.backbones.aggregator import decode_backbone_blob
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