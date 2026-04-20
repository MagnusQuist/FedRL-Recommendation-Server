import os
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.federated import BackboneDownload, BackboneUpload, RoundStatus, UploadAck
from app.db import get_db
from app.logger import logger

router = APIRouter(prefix="/federated")

CLIENTS_PER_ROUND = int(os.getenv("FEDERATED_CLIENTS_PER_ROUND", "2"))


def _get_aggregator(request: Request):
    """FastAPI dependency — retrieves the FLAggregator singleton from app state."""
    return request.app.state.aggregator


@router.get(
    "/status",
    response_model=RoundStatus,
    summary="Aggregation queue status",
)
async def backbone_status(
    aggregator=Depends(_get_aggregator),
):
    """Returns the current state of the FL aggregation queue."""
    current_version = aggregator.model_version

    return RoundStatus(
        current_version=current_version,
        queued_clients=aggregator.queued_client_ids(),
        total_rounds_completed=aggregator.rounds_completed(),
        clients_per_round=CLIENTS_PER_ROUND,
    )


@router.get(
    "/version",
    summary="Federated backbone version",
)
async def backbone_version(
    aggregator=Depends(_get_aggregator),
):
    """Return the current federated backbone version."""
    return { "version": aggregator.model_version }


@router.get(
    "/model",
    response_model=BackboneDownload,
    summary="Download current federated backbone",
    responses={304: {"description": "Client already has the latest version."}},
)
async def download_backbone(
    since: int = Query(0, ge=0, description="Client's current backbone version."),
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(_get_aggregator),
):
    """Return the current federated backbone if a newer version exists."""
    latest = await aggregator.get_current_version(db)

    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No backbone found. Run the seed script first.",
        )

    if latest.version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    return BackboneDownload(
        version=latest.version,
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
    request: Request,
    payload: BackboneUpload,
    db: AsyncSession = Depends(get_db),
    aggregator=Depends(_get_aggregator),
):
    """Accept a backbone weight upload from a Raspberry Pi client."""
    logger.info(
        "Received upload from client_id='%s' backbone_version=%d n_k=%d",
        payload.client_id,
        payload.backbone_version,
        payload.interaction_count,
    )

    latest = await aggregator.get_current_version(db)
    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No backbone found. Run the seed script first.",
        )

    if payload.backbone_version < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="backbone_version must be >= 1.",
        )

    # The payload includes a base64-encoded, gzip-compressed JSON blob of the
    # backbone weight tensors (matching GET /federated/model). Decode it here
    # before passing on to the aggregator.
    weights_dict = payload.backbone_weights
    full_request_size_bytes = len(await request.body())
    payload_blob = (
        payload.backbone_weights
        if isinstance(payload.backbone_weights, str)
        else json.dumps(payload.backbone_weights, separators=(",", ":"))
    )
    if isinstance(weights_dict, str):
        from app.backbones.aggregator import decode_backbone_blob
        weights_dict = decode_backbone_blob(weights_dict)

    round_triggered, queued = await aggregator.enqueue(
        client_id=payload.client_id,
        backbone_version=payload.backbone_version,
        interaction_count=payload.interaction_count,
        weights_dict=weights_dict,
        payload_blob=payload_blob,
        full_request_size_bytes=full_request_size_bytes,
        db=db,
    )

    return UploadAck(
        status="queued",
        client_id=payload.client_id,
        queued_clients=queued,
        round_triggered=round_triggered,
    )
