from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.logger import logger

from app.db.seed_backbone import SUPPORTED_ALGORITHMS


# ---------------------------------------------------------------------------
# The canonical set of backbone parameter keys.
# Any upload containing keys outside this set is rejected — this is the
# server-side enforcement of the privacy constraint that ensures local head
# weights are never transmitted.
# ---------------------------------------------------------------------------
BACKBONE_PARAM_KEYS = frozenset({
    "backbone.0.weight",   # Linear(28, 64) weights  — shape (64, 28)
    "backbone.0.bias",     # Linear(28, 64) bias     — shape (64,)
    "backbone.2.weight",   # Linear(64, 32) weights  — shape (32, 64)
    "backbone.2.bias",     # Linear(64, 32) bias     — shape (32,)
})


class GlobalBackboneVersionRead(BaseModel):
    """Pydantic mirror of ORM ``GlobalBackboneVersion`` in ``app.api.schemas.backbone``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    version: int = Field(..., description="Monotonic backbone round version for this algorithm.")
    weights_blob: str = Field(
        ...,
        description="gzip-compressed, base64-encoded JSON of backbone weight arrays.",
    )
    algorithm: str = Field(
        ...,
        max_length=10,
        description="Algorithm this backbone belongs to: 'ts'.",
    )
    client_count: int = Field(..., ge=0, description="Clients whose uploads contributed to this round.")
    total_interactions: int = Field(
        ...,
        ge=0,
        description="Sum of n_k across all contributing clients for this round.",
    )
    created_at: datetime


class BackboneUpload(BaseModel):
    """
    Payload sent by a Raspberry Pi client to POST /fl/upload.

    backbone_weights is the gzip-compressed, base64-encoded JSON representation
    of the backbone parameter tensors (matching what GET /fl/model returns).

    Version 1 is the initial seeded global backbone for each algorithm.
    """
    client_id: str = Field(
        ...,
        description="Arbitrary client identifier — no auth required.",
    )
    backbone_version: int = Field(
        ...,
        ge=1,
        description="The backbone version this upload was trained on top of.",
    )
    interaction_count: int = Field(
        ...,
        gt=0,
        description="n_k — interactions logged since the last upload.",
    )
    algorithm: str = Field(..., description="Training algorithm used by the client.")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        if v not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{v}'. Supported algorithms: {SUPPORTED_ALGORITHMS}"
            )
        return v
    
    backbone_weights: str = Field(
        ...,
        description="Backbone weights as a base64-encoded gzip string (matches GET /fl/model).",
    )

    @field_validator("backbone_weights", mode="before")
    @classmethod
    def validate_backbone_blob(cls, v: str | dict[str, list]) -> str | dict[str, list]:
        logger.info("Trying to validate backbone blob")
        """Validate that backbone_weights decodes to a valid backbone parameter dict."""
        if isinstance(v, str):
            try:
                from app.fl.aggregator import decode_backbone_blob as _decode

                decoded = _decode(v)
            except Exception as e:
                logger.error(e)
                raise ValueError(
                    "Invalid backbone_weights: expected a base64-encoded gzip blob of JSON. "
                    "See GET /fl/model for the expected encoding."
                ) from e
        elif isinstance(v, dict):
            decoded = v
        else:
            return v

        received = frozenset(decoded.keys())
        unexpected = received - BACKBONE_PARAM_KEYS
        missing = BACKBONE_PARAM_KEYS - received

        if unexpected:
            raise ValueError(
                f"Upload rejected — unexpected parameter keys detected: {sorted(unexpected)}. "
                "Only backbone parameters may be transmitted."
            )
        if missing:
            raise ValueError(
                f"Upload rejected — missing expected backbone keys: {sorted(missing)}."
            )

        return v


class BackboneDownload(BaseModel):
    """
    Response body for GET /backbone/model when a backbone is available.

    Subset of ``GlobalBackboneVersionRead``: ``backbone_weights`` is the API
    name for ORM ``weights_blob``.
    """

    version: int = Field(..., ge=1)
    algorithm: Literal["ts"]
    client_count: int = Field(..., ge=0)
    total_interactions: int = Field(..., ge=0)
    backbone_weights: str


class UploadAck(BaseModel):
    """Response body for a successfully queued upload."""
    status: str
    client_id: str
    queued_clients: int = Field(..., ge=0)
    round_triggered: bool


class RoundStatus(BaseModel):
    """Response body for GET /backbone/status — useful for debugging during experiments."""
    current_version: int = Field(
        ...,
        ge=0,
        description="Latest stored backbone version; 0 if none seeded yet.",
    )
    algorithm: Literal["ts"]
    queued_clients: list[str]
    total_rounds_completed: int = Field(..., ge=0)
    min_clients_per_round: int = Field(..., ge=1)
    round_timeout_seconds: int = Field(..., ge=1)