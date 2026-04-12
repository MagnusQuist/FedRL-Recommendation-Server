"""Pydantic schemas for the centralized training endpoints."""

from pydantic import BaseModel, Field


class InteractionUpload(BaseModel):
    """Request body for POST /centralized/interactions."""
    client_id: str = Field(..., description="Client identifier.")
    algorithm: str = Field(..., description="Algorithm used by the client (e.g. 'ts').")
    count: int = Field(..., gt=0, description="Number of interaction tuples in this batch.")
    data: str = Field(
        ...,
        description="gzip+base64 encoded JSON array of interaction tuples.",
    )


class InteractionAck(BaseModel):
    """Response body for POST /centralized/interactions."""
    accepted: bool
    server_model_version: int


class CentralizedModelDownload(BaseModel):
    """Response body for GET /centralized/model."""
    model_version: int = Field(..., ge=0)
    backbone_weights: str = Field(
        ...,
        description="gzip+base64 encoded backbone state dict.",
    )
    head_weights: dict[str, str] = Field(
        ...,
        description="Per-head gzip+base64 encoded state dicts (item, price, nudge).",
    )
