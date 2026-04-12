"""Router for the centralized training endpoints.

POST /centralized/interactions — receive raw interaction tuples
GET  /centralized/model        — download the latest centralized model
"""

from app.logger import logger

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response

from app.api.schemas.centralized import (
    CentralizedModelDownload,
    InteractionAck,
    InteractionUpload,
)
from app.db.seed_backbone import FEDERATED_ALGORITHM

router = APIRouter(prefix="/centralized")


def _get_centralized_service(request: Request):
    return request.app.state.centralized_service


def _validate_algorithm(algorithm: str) -> str:
    if algorithm != FEDERATED_ALGORITHM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported algorithm '{algorithm}'. "
                f"Only '{FEDERATED_ALGORITHM}' is supported."
            ),
        )
    return algorithm


@router.post(
    "/interactions",
    response_model=InteractionAck,
    summary="Upload interaction tuples for centralized training",
)
async def upload_interactions(payload: InteractionUpload, request: Request):
    """Receive a batch of raw interaction tuples from a centralized-mode client."""
    _validate_algorithm(payload.algorithm)

    service = _get_centralized_service(request)

    try:
        new_version = await service.process_interactions(
            client_id=payload.client_id,
            algorithm=payload.algorithm,
            count=payload.count,
            data=payload.data,
        )
    except Exception as e:
        logger.exception("Centralized interaction processing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process interactions: {e}",
        ) from e

    return InteractionAck(accepted=True, server_model_version=new_version)


@router.get(
    "/model",
    response_model=CentralizedModelDownload,
    summary="Download the current centralized model (backbone + heads)",
    responses={304: {"description": "Client already has the latest version."}},
)
async def download_centralized_model(
    since: int = Query(0, ge=0, description="Client's current model version."),
    algorithm: str = Query("ts", description="Algorithm to download."),
    request: Request = None,
):
    """Return the centralized backbone and head weights if a newer version exists."""
    _validate_algorithm(algorithm)

    service = _get_centralized_service(request)

    if service.model_version <= since:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED)

    snapshot = service.get_model_snapshot()
    return CentralizedModelDownload(**snapshot)
