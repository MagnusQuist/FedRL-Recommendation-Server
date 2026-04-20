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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.db.models.federated import FederatedBackboneVersion
from app.db.models.training_payload_log import TrainingPayloadLog
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
    payload_blob: str
    full_request_size_bytes: int
    received_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Pure CPU-bound helpers (safe to run in a worker thread)
# ---------------------------------------------------------------------------
def _fedavg_and_serialize(
    eligible: list[QueuedUpload], n_total: int
) -> str:
    """
    Execute FedAvg over ``eligible`` uploads and return the gzip+base64 blob
    suitable for direct insertion into ``FederatedBackboneVersion.weights_blob``.
    """
    aggregated: dict[str, np.ndarray] = {}
    param_keys = list(eligible[0].weights.keys())

    for key in param_keys:
        aggregated[key] = sum(
            (u.interaction_count / n_total) * u.weights[key]
            for u in eligible
        )

    weights_json = {k: v.tolist() for k, v in aggregated.items()}
    compressed = gzip.compress(json.dumps(weights_json).encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


# ---------------------------------------------------------------------------
# Aggregator (singleton, held in app state)
# ---------------------------------------------------------------------------
class FLAggregator:
    def __init__(self) -> None:
        self._queue: dict[str, QueuedUpload] = {}
        self._rounds_completed: int = 0
        # Cached latest persisted version. Updated on startup via
        # ``try_load_persisted_state`` and after every successful FedAvg round.
        self.model_version: int = 0
        self._lock = asyncio.Lock()

    # ── Initialisation ──────────────────────────────────────────────────────

    async def try_load_persisted_state(self) -> bool:
        """
        Returns True if any persisted state was found.
        """
        try:
            async with AsyncSessionLocal() as db:
                latest_version = (
                    await db.execute(select(func.max(FederatedBackboneVersion.version)))
                ).scalar()

                completed_rounds = (
                    await db.execute(
                        select(func.count(FederatedBackboneVersion.id)).where(
                            FederatedBackboneVersion.client_count > 0
                        )
                    )
                ).scalar() or 0

            if latest_version is None:
                logger.info("FL aggregator: no persisted state found.")
                return False

            self.model_version = int(latest_version)
            self._rounds_completed = int(completed_rounds)

            logger.info(
                "FL aggregator state restored: version=%d rounds_completed=%d",
                self.model_version,
                self._rounds_completed,
            )
            return True

        except Exception:
            logger.exception(
                "Failed to load persisted FL aggregator state — starting fresh."
            )
            return False

    # ── Public API ──────────────────────────────────────────────────────────

    async def enqueue(
        self,
        client_id: str,
        backbone_version: int,
        interaction_count: int,
        weights_dict: dict[str, list],
        payload_blob: str,
        full_request_size_bytes: int,
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
                payload_blob=payload_blob,
                full_request_size_bytes=full_request_size_bytes,
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

        # Offload the CPU-bound reduce + gzip + base64 to a worker thread,
        # backbone retraining in ``asyncio.to_thread``. Keeps the API event
        # loop responsive when a round is triggered.
        train_start = time.perf_counter()
        blob = await asyncio.to_thread(_fedavg_and_serialize, eligible, n_total)
        training_time_seconds = time.perf_counter() - train_start

        next_version = await self._next_version(db)

        new_backbone = FederatedBackboneVersion(
            version=next_version,
            weights_blob=blob,
            client_count=len(eligible),
            total_interactions=n_total,
            training_time_seconds=training_time_seconds,
        )
        db.add(new_backbone)
        await db.flush()

        for upload in eligible:
            size_bytes = len(upload.payload_blob.encode("utf-8"))
            size_kb = size_bytes / 1024
            size_mb = size_bytes / (1024 * 1024)
            db.add(
                TrainingPayloadLog(
                    client_id=upload.client_id,
                    payload_blob=upload.payload_blob,
                    payload_size_bytes=size_bytes,
                    payload_size_kb=size_kb,
                    payload_size_mb=size_mb,
                    full_request_size_bytes=upload.full_request_size_bytes,
                    federated_model_version_id=new_backbone.id,
                )
            )
            logger.info(
                "Federated round payload logged: version=%d client_id='%s' "
                "payload_size=%.2f KB (%.4f MB) full_request_size=%d bytes",
                next_version,
                upload.client_id,
                size_kb,
                size_mb,
                upload.full_request_size_bytes,
            )

        await db.commit()
        await db.refresh(new_backbone)

        self.model_version = new_backbone.version
        self._rounds_completed += 1
        logger.info(
            "FedAvg round complete — version=%d clients=%d interactions=%d "
            "training_time=%.3fs rounds_completed=%d",
            new_backbone.version,
            len(eligible),
            n_total,
            training_time_seconds,
            self._rounds_completed,
        )

        self._queue.clear()


def decode_backbone_blob(blob: str) -> dict[str, list]:
    logger.info("Decoding backbone weights")
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
