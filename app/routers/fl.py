from fastapi import APIRouter

router = APIRouter()


# ---------------------------------------------------------------------------
# Federated Learning endpoints — aggregation logic not yet implemented.
# Stubs are in place so the router is registered and the API contract is
# visible in the OpenAPI docs. Implementation tracked in a separate task.
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    status_code=202,
    summary="[STUB] Upload backbone weights",
)
async def upload_backbone():
    """
    Accepts a backbone weight upload from a Raspberry Pi client.

    Expected body (not yet validated):
        {
            "backbone_weights": { ... },   # gzip-compressed, base64-encoded
            "interaction_count": int,       # n_k since last upload
            "client_id": str,
            "backbone_version": int
        }

    Returns 202 Accepted while the payload is queued for the next FedAvg round.
    """
    return {"status": "stub — aggregation not yet implemented"}


@router.get(
    "/model",
    summary="[STUB] Download current global backbone",
)
async def download_backbone(since: int = 0, algorithm: str = "ts"):
    """
    Returns the current global backbone if a newer version exists.

    Query params:
        since     — client's current backbone version (returns 304 if up to date)
        algorithm — 'ts' or 'dqn'

    Returns 200 with weights, or 304 Not Modified.
    """
    return {"status": "stub — aggregation not yet implemented", "since": since, "algorithm": algorithm}