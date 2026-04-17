"""Router for the centralized training endpoints.

POST /centralized/interactions — receive raw interaction tuples
GET  /centralized/model        — download the latest centralized model
"""

from fastapi import APIRouter, HTTPException, Query, Request, status, Depends
from fastapi.responses import Response

from app.api.schemas.centralized import (
    CentralizedModelDownload,
    InteractionAck,
    InteractionUpload,
    CentralizedTrainingStatus,
)
from app.backbones.centralized import CentralizedService
from app.logger import logger
import os

CENTRALIZED_CLIENTS_PER_ROUND = int(os.getenv("CENTRALIZED_CLIENTS_PER_ROUND", "2"))

router = APIRouter(prefix="/centralized")


def _get_centralized_service(request: Request):
    """FastAPI dependency — retrieves the CentralizedService singleton from app state."""
    return request.app.state.centralized_service


@router.post(
    "/interactions",
    response_model=InteractionAck,
    summary="Upload interaction tuples for centralized training",
)
async def upload_interactions(
    payload: InteractionUpload, 
    service: CentralizedService = Depends(_get_centralized_service),
):
    """Receive a batch of raw interaction tuples from a centralized-mode client.

    The upload is buffered; a training round runs only when exactly
    ``CENTRALIZED_CLIENTS_PER_ROUND`` unique clients have uploaded.
    """
    try:
        model_version, round_triggered, queued_clients = await service.process_interactions(
            client_id=payload.client_id,
            count=payload.count,
            data=payload.data,
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
    """Return the centralized backbone and head weights if a newer version exists."""
    if service.model_version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    snapshot = service.get_model_snapshot()
    return CentralizedModelDownload(**snapshot)

@router.get(
    "/version",
    summary="Centralized model version",
)
async def centralized_model_version(
    service: CentralizedService = Depends(_get_centralized_service),
):
    """Return the current centralized model version."""
    return service.model_version


@router.get(
    "/status",
    response_model=CentralizedTrainingStatus,
    summary="Centralized training queue status",
)
async def centralized_training_status(
    service: CentralizedService = Depends(_get_centralized_service),
):
    """Returns the current state of the centralized training queue."""
    return CentralizedTrainingStatus(
        current_version=service.model_version,
        queued_clients=len(service._pending_clients),
        total_rounds_completed=service._rounds_completed,
        pool_size=len(service._tuple_pool),
        clients_per_round=CENTRALIZED_CLIENTS_PER_ROUND,
    )