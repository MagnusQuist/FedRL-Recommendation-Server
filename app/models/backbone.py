from pydantic import BaseModel, Field, field_validator


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


class BackboneUpload(BaseModel):
    """
    Payload sent by a Raspberry Pi client to POST /fl/upload.

    backbone_weights is a dict mapping each parameter key to a nested list
    of floats (the raw weight tensor serialised as JSON). The payload is
    expected to arrive gzip-compressed and base64-encoded by the client,
    but FastAPI decompresses it before this schema is applied.
    """
    client_id: str = Field(..., description="Arbitrary client identifier — no auth required.")
    backbone_version: int = Field(..., ge=0, description="The backbone version this upload was trained on top of.")
    interaction_count: int = Field(..., gt=0, description="n_k — interactions logged since the last upload.")
    algorithm: str = Field("ts", pattern="^(ts|dqn)$", description="'ts' or 'dqn'.")
    backbone_weights: dict[str, list] = Field(
        ...,
        description="Backbone parameter tensors serialised as nested lists of floats.",
    )

    @field_validator("backbone_weights")
    @classmethod
    def validate_backbone_keys(cls, v: dict) -> dict:
        """
        Reject any payload whose keys are not exactly the known backbone params.
        This is the server-side privacy enforcement: if a client accidentally
        (or maliciously) includes local head parameters, the upload is refused.
        """
        received = frozenset(v.keys())
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
    Response body for GET /fl/model when a newer backbone is available.

    backbone_weights is the gzip-compressed, base64-encoded JSON representation
    of the backbone parameter tensors. Once decoded and decompressed, it
    matches the structure that clients originally uploaded.
    """
    version: int
    algorithm: str
    client_count: int
    total_interactions: int
    backbone_weights: str


class UploadAck(BaseModel):
    """Response body for a successfully queued upload."""
    status: str
    client_id: str
    queued_clients: int
    round_triggered: bool


class RoundStatus(BaseModel):
    """Response body for GET /fl/status — useful for debugging during experiments."""
    current_version: int
    algorithm: str
    queued_clients: list[str]
    total_rounds_completed: int
    min_clients_per_round: int
    round_timeout_seconds: int