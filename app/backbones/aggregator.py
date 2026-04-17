"""
FL Aggregation Service
======================
Manages in-memory upload queues and executes FedAvg rounds for the federated
backbone.

Design decisions reflected here:
- Uploads are queued in memory and keyed by client_id.
- If the same client uploads twice before a round triggers, the newer upload
  replaces the older one (queue size does not grow).
- A round triggers when exactly ``FL_MIN_CLIENTS_PER_ROUND`` unique clients
  have uploaded. There is no timeout — strict batch semantics, so the FL round
  and the centralized training batch share an identical client-count constraint
  (fair experimental comparison).
- FedAvg: w* = Σ (n_k / n_total) * w_k — pure NumPy, no PyTorch required.
- The aggregated backbone is persisted to PostgreSQL with monotonic versioning.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.federated import FederatedBackboneVersion
from app.logger import logger

# ---------------------------------------------------------------------------
# Configuration — overridable via environment variables
# ---------------------------------------------------------------------------
CLIENTS_PER_ROUND = int(os.getenv("FEDERATED_CLIENTS_PER_ROUND", "2"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class QueuedUpload:
    client_id: str
    backbone_version: int
    interaction_count: int
    weights: dict[str, np.ndarray]
    received_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Aggregator (singleton, held in app state)
# ---------------------------------------------------------------------------
class FLAggregator:
    def __init__(self) -> None:
        self._queue: dict[str, QueuedUpload] = {}
        self._rounds_completed: int = 0
        self._lock = asyncio.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    async def enqueue(
        self,
        client_id: str,
        backbone_version: int,
        interaction_count: int,
        weights_dict: dict[str, list],
        db: AsyncSession,
    ) -> tuple[bool, int]:
        """
        Add or replace a client's upload in the queue.

        Returns ``(round_triggered, queued_client_count)``. If a round triggers,
        FedAvg is run and the result persisted to Postgres.
        """
        weights = {k: np.array(v, dtype=np.float32) for k, v in weights_dict.items()}

        async with self._lock:
            self._queue[client_id] = QueuedUpload(
                client_id=client_id,
                backbone_version=backbone_version,
                interaction_count=interaction_count,
                weights=weights,
            )

            queued = len(self._queue)
            logger.info(
                "Queued upload from '%s' — %d/%d clients ready (n_k=%d)",
                client_id,
                queued,
                CLIENTS_PER_ROUND,
                interaction_count,
            )

            # The lock guarantees we never jump from N-1 to N+1: every enqueue
            # increments by 1 or replaces an existing entry, and the check
            # happens before the next enqueue can run. Any queue size above N
            # indicates a programmer error.
            assert queued <= CLIENTS_PER_ROUND, (
                f"Federated queue overshot the configured batch size: "
                f"{queued} > {CLIENTS_PER_ROUND}. This should be impossible "
                "with the aggregator lock held."
            )

            triggered = queued == CLIENTS_PER_ROUND
            if triggered:
                await self._run_fedavg(db)

            return triggered, len(self._queue)

    async def get_current_version(
        self,
        db: AsyncSession,
    ) -> Optional[FederatedBackboneVersion]:
        """Return the latest persisted federated backbone."""
        result = await db.execute(
            select(FederatedBackboneVersion)
            .order_by(FederatedBackboneVersion.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def queued_client_ids(self) -> list[str]:
        return list(self._queue.keys())

    def rounds_completed(self) -> int:
        return self._rounds_completed

    def metrics_snapshot(self) -> dict[str, Any]:
        """
        Lightweight snapshot of in-memory aggregation state for debugging.

        Does not expose model weights.
        """
        queued_uploads = list(self._queue.values())
        queued_client_ids = [u.client_id for u in queued_uploads]
        queued_client_count = len(queued_client_ids)

        queued_total_interactions = sum(u.interaction_count for u in queued_uploads)
        queued_avg_interactions = (
            queued_total_interactions / queued_client_count
            if queued_client_count
            else None
        )

        oldest_received = min((u.received_at for u in queued_uploads), default=None)
        oldest_upload_age_seconds = (
            time.monotonic() - oldest_received if oldest_received is not None else None
        )

        return {
            "queued_client_ids": queued_client_ids,
            "queued_client_count": queued_client_count,
            "queued_total_interactions": queued_total_interactions,
            "queued_avg_interactions": queued_avg_interactions,
            "queued_oldest_upload_age_seconds": oldest_upload_age_seconds,
            "rounds_completed": self._rounds_completed,
            "clients_per_round": CLIENTS_PER_ROUND,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    async def _next_version(self, db: AsyncSession) -> int:
        latest = await self.get_current_version(db)
        return 1 if latest is None else latest.version + 1

    async def _run_fedavg(self, db: AsyncSession) -> None:
        """
        Execute FedAvg over all queued uploads and persist the result.

        w* = Σ (n_k / n_total) * w_k
        """
        if not self._queue:
            logger.warning("FedAvg triggered but queue is empty.")
            return

        eligible = list(self._queue.values())
        n_total = sum(u.interaction_count for u in eligible)

        if n_total <= 0:
            logger.warning(
                "FedAvg aborted because total interactions is %d.",
                n_total,
            )
            return

        # Optional safety check: all uploads should target the same backbone version.
        base_versions = {u.backbone_version for u in eligible}
        if len(base_versions) > 1:
            logger.warning(
                "FedAvg includes mixed base versions: %s",
                sorted(base_versions),
            )

        aggregated: dict[str, np.ndarray] = {}
        param_keys = list(eligible[0].weights.keys())

        for key in param_keys:
            aggregated[key] = sum(
                (u.interaction_count / n_total) * u.weights[key]
                for u in eligible
            )

        weights_json = {k: v.tolist() for k, v in aggregated.items()}
        compressed = gzip.compress(json.dumps(weights_json).encode("utf-8"))
        blob = base64.b64encode(compressed).decode("utf-8")

        next_version = await self._next_version(db)

        new_backbone = FederatedBackboneVersion(
            version=next_version,
            weights_blob=blob,
            client_count=len(eligible),
            total_interactions=n_total,
        )
        db.add(new_backbone)
        await db.flush()
        await db.commit()
        await db.refresh(new_backbone)

        self._rounds_completed += 1
        logger.info(
            "FedAvg round complete — version=%d clients=%d interactions=%d rounds_completed=%d",
            new_backbone.version,
            len(eligible),
            n_total,
            self._rounds_completed,
        )

        self._queue.clear()


def decode_backbone_blob(blob: str) -> dict[str, list]:
    logger.info("Decoding backbone weights")
    if not isinstance(blob, str):
        raise ValueError("backbone_weights must be a base64-encoded string")

    try:
        compressed_bytes = base64.b64decode(blob)
    except Exception as e:
        raise ValueError("Invalid base64 encoding in backbone_weights") from e

    try:
        json_bytes = gzip.decompress(compressed_bytes)
    except Exception as e:
        raise ValueError("Invalid gzip payload in backbone_weights") from e

    try:
        decoded: Any = json.loads(json_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("Invalid JSON in decompressed backbone_weights") from e

    if not isinstance(decoded, dict):
        raise ValueError("Decoded backbone_weights must be a JSON object")

    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ValueError("Decoded backbone_weights contains a non-string parameter name")
        if not isinstance(value, list):
            raise ValueError(
                f"Decoded backbone_weights parameter '{key}' must map to a list"
            )

    return decoded
