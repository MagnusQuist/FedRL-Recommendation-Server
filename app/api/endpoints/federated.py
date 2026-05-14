from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import AggregationEvent
from app.ml.aggregation import CLIENTS_PER_ROUND, FLAggregator, decode_backbone_blob
from app.schemas.training import BackboneDownload, BackboneUpload, RoundStatus, UploadAck
from app.logging import logger

router = APIRouter(prefix="/federated")


def _get_aggregator(request: Request) -> FLAggregator:
    return request.app.state.aggregator


@router.get("/status", response_model=RoundStatus, summary="Aggregation queue status")
async def backbone_status(aggregator: FLAggregator = Depends(_get_aggregator)):
    return RoundStatus(
        current_version=aggregator.model_version,
        queued_clients=aggregator.queued_client_ids(),
        total_rounds_completed=aggregator.rounds_completed(),
        clients_per_round=CLIENTS_PER_ROUND,
    )


@router.get("/version", summary="Federated backbone version")
async def backbone_version(aggregator: FLAggregator = Depends(_get_aggregator)):
    return {"version": aggregator.model_version}


@router.get(
    "/model",
    response_model=BackboneDownload,
    summary="Download current federated backbone",
    responses={304: {"description": "Client already has the latest version."}},
)
async def download_backbone(
    since: int = Query(0, ge=0, description="Client's current backbone version."),
    db: AsyncSession = Depends(get_db),
    aggregator: FLAggregator = Depends(_get_aggregator),
):
    latest = await aggregator.get_current_version(db)

    if latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No backbone found. Run the seed script first.",
        )

    if latest.version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    round_metrics_result = await db.execute(
        select(AggregationEvent)
        .where(AggregationEvent.model_version_after == str(latest.version))
        .order_by(AggregationEvent.timestamp.desc())
        .limit(1)
    )
    latest_round_metrics = round_metrics_result.scalar_one_or_none()

    return BackboneDownload(
        version=latest.version,
        client_count=latest_round_metrics.num_clients_in_round if latest_round_metrics else 0,
        total_interactions=latest_round_metrics.total_interactions if latest_round_metrics else 0,
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
    aggregator: FLAggregator = Depends(_get_aggregator),
):
    logger.info(
        "Received upload from client_id='%s' backbone_version=%d n_k=%d",
        payload.client_id, payload.backbone_version, payload.interaction_count,
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

    weights_dict = payload.backbone_weights
    if isinstance(weights_dict, str):
        weights_dict = decode_backbone_blob(weights_dict)

    round_triggered, queued = await aggregator.enqueue(
        client_id=payload.client_id,
        backbone_version=payload.backbone_version,
        interaction_count=payload.interaction_count,
        weights_dict=weights_dict,
        db=db,
    )

    return UploadAck(
        status="queued",
        client_id=payload.client_id,
        queued_clients=queued,
        round_triggered=round_triggered,
    )
