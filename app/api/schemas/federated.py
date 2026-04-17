from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.logger import logger


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


class FederatedBackboneVersionRead(BaseModel):
    """Pydantic mirror of ORM ``FederatedBackboneVersion``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    version: int = Field(..., description="Monotonic federated backbone round version.")
    weights_blob: str = Field(
        ...,
        description="gzip-compressed, base64-encoded JSON of backbone weight arrays.",
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
    Payload sent by a Raspberry Pi client to POST /federated/model.

    backbone_weights is the gzip-compressed, base64-encoded JSON representation
    of the backbone parameter tensors (matching what GET /federated/model returns).

    Version 1 is the initial seeded federated backbone.
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
    backbone_weights: str = Field(
        ...,
        description="Backbone weights as a base64-encoded gzip string (matches GET /federated/model).",
    )

    @field_validator("backbone_weights", mode="before")
    @classmethod
    def validate_backbone_blob(cls, v: str | dict[str, list]) -> str | dict[str, list]:
        """Validate that backbone_weights decodes to a valid backbone parameter dict."""
        logger.info("Trying to validate backbone blob")
        if isinstance(v, str):
            try:
                from app.backbones.aggregator import decode_backbone_blob as _decode

                decoded = _decode(v)
            except Exception as e:
                logger.error(e)
                raise ValueError(
                    "Invalid backbone_weights: expected a base64-encoded gzip blob of JSON. "
                    "See GET /federated/model for the expected encoding."
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
    Response body for GET /federated/model when a backbone is available.

    Subset of ``FederatedBackboneVersionRead``: ``backbone_weights`` is the API
    name for ORM ``weights_blob``.
    """

    version: int = Field(..., ge=1)
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
    """Response body for GET /federated/status — useful for debugging during experiments."""
    current_version: int = Field(
        ...,
        ge=0,
        description="Latest stored backbone version; 0 if none seeded yet.",
    )
    queued_clients: list[str]
    total_rounds_completed: int = Field(..., ge=0)
    clients_per_round: int = Field(
        ...,
        ge=1,
        description="Exact number of unique clients required to trigger a FedAvg round.",
    )
