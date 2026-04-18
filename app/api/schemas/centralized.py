"""Pydantic schemas for the centralized training endpoints."""

from pydantic import BaseModel, Field


class InteractionUpload(BaseModel):
    """Request body for POST /centralized/interactions."""
    client_id: str = Field(..., description="Client identifier.")
    count: int = Field(..., gt=0, description="Number of interaction tuples in this batch.")
    data: str = Field(
        ...,
        description="gzip+base64 encoded JSON array of interaction tuples.",
    )


class InteractionAck(BaseModel):
    """Response body for POST /centralized/interactions."""
    accepted: bool
    server_model_version: int = Field(
        ...,
        ge=0,
        description=(
            "Latest persisted centralized model version. Only advances when a "
            "training round was triggered by this upload."
        ),
    )
    round_triggered: bool = Field(
        ...,
        description=(
            "True when this upload completed a batch of "
            "CENTRALIZED_CLIENTS_PER_ROUND unique clients and a training "
            "round ran on the server."
        ),
    )
    queued_clients: int = Field(
        ...,
        ge=0,
        description=(
            "Number of unique clients currently buffered for the next "
            "centralized training round. Resets to 0 immediately after a "
            "round runs."
        ),
    )


class CentralizedTrainingStatus(BaseModel):
    """Response body for GET /centralized/status."""
    current_version: int = Field(
        ...,
        ge=0,
        description="Latest persisted centralized model version; 0 if none seeded yet.",
    )
    queued_clients: int = Field(
        ...,
        ge=0,
        description=(
            "Number of unique clients currently buffered for the next "
            "centralized training round. Resets to 0 immediately after a "
            "round runs."
        ),
    )
    total_rounds_completed: int = Field(
        ...,
        ge=0,
        description="Number of centralized training rounds completed.",
    )
    pool_size: int = Field(
        ...,
        ge=0,
        description="Number of interaction tuples currently in the pool.",
    )
    clients_per_round: int = Field(
        ...,
        ge=2,
        description="Exact number of unique clients required to trigger a centralized training round.",
    )


class CentralizedModelDownload(BaseModel):
    """Response body for GET /centralized/model."""
    
    version: int = Field(..., ge=0)
    backbone_weights: str = Field(
        ...,
        description="gzip+base64 encoded backbone state dict.",
    )
    head_weights: dict[str, str] = Field(
        ...,
        description="Per-head gzip+base64 encoded state dicts (item, price, nudge).",
    )
