import os

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from app.ml.centralized_training import CentralizedService
from app.schemas.training import (
    CentralizedModelDownload,
    CentralizedTrainingStatus,
    InteractionAck,
    InteractionUpload,
)
from app.logging import logger

CLIENTS_PER_ROUND = int(os.getenv("CENTRALIZED_CLIENTS_PER_ROUND", "2"))

router = APIRouter(prefix="/centralized")


def _get_centralized_service(request: Request) -> CentralizedService:
    return request.app.state.centralized_service


@router.post("/interactions", response_model=InteractionAck, summary="Upload interaction tuples for centralized training")
async def upload_interactions(
    client_interactions: InteractionUpload,
    centralized_service: CentralizedService = Depends(_get_centralized_service),
):
    try:
        model_version, round_triggered, queued_clients = await centralized_service.process_interactions(
            client_id=client_interactions.client_id,
            count=client_interactions.count,
            data=client_interactions.data,
        )
    except Exception as e:
        logger.exception("Centralized interaction processing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process interactions: {e}",
        ) from e

    return InteractionAck(
        accepted=True,
        server_model_version=model_version,
        round_triggered=round_triggered,
        queued_clients=queued_clients,
    )


@router.get(
    "/model",
    response_model=CentralizedModelDownload,
    summary="Download the current centralized model (backbone + heads)",
    responses={304: {"description": "Client already has the latest version."}},
)
async def download_centralized_model(
    since: int = Query(0, ge=0, description="Client's current model version."),
    service: CentralizedService = Depends(_get_centralized_service),
):
    if service.model_version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    snapshot = service.get_model_snapshot()
    return CentralizedModelDownload(**snapshot)


@router.get("/version", summary="Centralized model version")
async def centralized_model_version(service: CentralizedService = Depends(_get_centralized_service)):
    return {"version": service.model_version}


@router.get("/status", response_model=CentralizedTrainingStatus, summary="Centralized training queue status")
async def centralized_training_status(service: CentralizedService = Depends(_get_centralized_service)):
    return CentralizedTrainingStatus(
        current_version=service.model_version,
        queued_clients=len(service._pending_uploads),
        total_rounds_completed=service._rounds_completed,
        pool_size=len(service._tuple_pool),
        clients_per_round=CLIENTS_PER_ROUND,
    )
