from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.logging import logger


# ---------------------------------------------------------------------------
# The canonical set of backbone parameter keys.
# Any upload containing keys outside this set is rejected — this is the
# server-side enforcement of the privacy constraint that ensures local head
# weights are never transmitted.
# ---------------------------------------------------------------------------
BACKBONE_PARAM_KEYS = frozenset({
    "backbone.0.weight",   # Linear(16, 64) weights  — shape (64, 16)
    "backbone.0.bias",     # Linear(16, 64) bias     — shape (64,)
    "backbone.2.weight",   # Linear(64, 32) weights  — shape (32, 64)
    "backbone.2.bias",     # Linear(64, 32) bias     — shape (32,)
})


# ── Federated schemas ────────────────────────────────────────────────────────

class FederatedModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: int = Field(..., description="Monotonic federated backbone round version.")
    weights_blob: str = Field(
        ...,
        description="gzip-compressed, base64-encoded JSON of backbone weight arrays.",
    )
    created_at: datetime


class BackboneUpload(BaseModel):
    client_id: str = Field(..., description="Arbitrary client identifier — no auth required.")
    backbone_version: int = Field(..., ge=1, description="The backbone version this upload was trained on top of.")
    interaction_count: int = Field(..., gt=0, description="n_k — interactions logged since the last upload.")
    backbone_weights: str = Field(
        ...,
        description="Backbone weights as a base64-encoded gzip string (matches GET /federated/model).",
    )

    @field_validator("backbone_weights", mode="before")
    @classmethod
    def validate_backbone_blob(cls, v: str | dict[str, list]) -> str | dict[str, list]:
        logger.info("Trying to validate backbone blob")
        if isinstance(v, str):
            try:
                from app.ml.aggregation import decode_backbone_blob as _decode
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
    version: int = Field(..., ge=1)
    client_count: int = Field(..., ge=0)
    total_interactions: int = Field(..., ge=0)
    backbone_weights: str


class UploadAck(BaseModel):
    status: str
    client_id: str
    queued_clients: int = Field(..., ge=0)
    round_triggered: bool


class RoundStatus(BaseModel):
    current_version: int = Field(..., ge=0, description="Latest stored backbone version; 0 if none seeded yet.")
    queued_clients: list[str]
    total_rounds_completed: int = Field(..., ge=0)
    clients_per_round: int = Field(..., ge=1, description="Exact number of unique clients required to trigger a FedAvg round.")


# ── Centralized schemas ──────────────────────────────────────────────────────

class InteractionUpload(BaseModel):
    client_id: str = Field(..., description="Client identifier.")
    count: int = Field(..., gt=0, description="Number of interaction tuples in this batch.")
    data: str = Field(..., description="gzip+base64 encoded JSON array of interaction tuples.")


class InteractionAck(BaseModel):
    accepted: bool
    server_model_version: int = Field(..., ge=0)
    round_triggered: bool
    queued_clients: int = Field(..., ge=0)


class CentralizedTrainingStatus(BaseModel):
    current_version: int = Field(..., ge=0, description="Latest persisted centralized model version; 0 if none seeded yet.")
    queued_clients: int = Field(..., ge=0)
    total_rounds_completed: int = Field(..., ge=0)
    pool_size: int = Field(..., ge=0)
    clients_per_round: int = Field(..., ge=2)


class CentralizedModelDownload(BaseModel):
    version: int = Field(..., ge=0)
    backbone_weights: str = Field(..., description="gzip+base64 encoded backbone state dict.")
    reward_predictor_weights: str = Field(..., description="gzip+base64 encoded reward predictor state dict.")
    head_weights: dict[str, str] = Field(..., description="Per-head gzip+base64 encoded state dicts (item, price, nudge).")
