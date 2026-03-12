"""
FL Aggregation Service
======================
Manages the in-memory upload queue and executes FedAvg rounds.

Design decisions reflected here:
- Uploads are queued in memory keyed by client_id. If the same client uploads
  twice before a round triggers, the newer upload replaces the older one.
- A round triggers when queued uploads >= min_clients_per_round OR
  round_timeout_seconds elapses since the first upload arrived in the queue.
- If round_timeout elapses but the queue is below min_clients_per_round,
  uploads are carried forward to the next round (not discarded).
- FedAvg: w* = Σ (n_k / n_total) * w_k — pure NumPy, no PyTorch required.
- The aggregated backbone is persisted to PostgreSQL and the queue is cleared.
"""

import asyncio
import base64
import gzip
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.backbone import GlobalBackboneVersion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — overridable via environment variables
# ---------------------------------------------------------------------------
MIN_CLIENTS_PER_ROUND = int(os.getenv("FL_MIN_CLIENTS_PER_ROUND", "2"))
ROUND_TIMEOUT_SECONDS = int(os.getenv("FL_ROUND_TIMEOUT_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class QueuedUpload:
    client_id: str
    backbone_version: int
    interaction_count: int
    algorithm: str
    weights: dict[str, np.ndarray]  # key → numpy array
    received_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Aggregator (singleton, held in app state)
# ---------------------------------------------------------------------------
class FLAggregator:
    def __init__(self) -> None:
        self._queue: dict[str, QueuedUpload] = {}
        self._round_start: Optional[float] = None
        self._rounds_completed: int = 0
        self._lock = asyncio.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    async def enqueue(
        self,
        client_id: str,
        backbone_version: int,
        interaction_count: int,
        algorithm: str,
        weights_dict: dict[str, list],
        db: AsyncSession,
    ) -> tuple[bool, int]:
        """
        Add or replace a client's upload in the queue.

        Returns (round_triggered, queued_client_count).
        If a round triggers, FedAvg is run and the result persisted to Postgres.
        """
        weights = {k: np.array(v, dtype=np.float32) for k, v in weights_dict.items()}

        async with self._lock:
            self._queue[client_id] = QueuedUpload(
                client_id=client_id,
                backbone_version=backbone_version,
                interaction_count=interaction_count,
                algorithm=algorithm,
                weights=weights,
            )

            # Record when the first upload of this round arrived
            if self._round_start is None:
                self._round_start = time.monotonic()

            queued = len(self._queue)
            logger.info(
                "Queued upload from '%s' — %d/%d clients ready (algorithm=%s, n_k=%d)",
                client_id, queued, MIN_CLIENTS_PER_ROUND, algorithm, interaction_count,
            )

            triggered = self._should_trigger()
            if triggered:
                await self._run_fedavg(algorithm, db)

        return triggered, len(self._queue)

    async def check_timeout(self, db: AsyncSession) -> bool:
        """
        Called periodically by the background timeout task.
        Triggers a round if round_timeout has elapsed and at least one upload
        is queued. Uploads below min_clients_per_round are carried forward.
        """
        async with self._lock:
            if not self._queue or self._round_start is None:
                return False

            elapsed = time.monotonic() - self._round_start
            if elapsed < ROUND_TIMEOUT_SECONDS:
                return False

            queued = len(self._queue)
            if queued < MIN_CLIENTS_PER_ROUND:
                logger.info(
                    "Round timeout elapsed but only %d/%d clients queued — "
                    "carrying uploads forward.",
                    queued, MIN_CLIENTS_PER_ROUND,
                )
                # Reset timer so we don't log this every second
                self._round_start = time.monotonic()
                return False

            # Timeout elapsed AND we have enough clients — trigger
            algorithm = self._majority_algorithm()
            await self._run_fedavg(algorithm, db)
            return True

    async def get_current_version(self, db: AsyncSession) -> Optional[GlobalBackboneVersion]:
        """Return the latest backbone row from Postgres, or None if none exists."""
        result = await db.execute(
            select(GlobalBackboneVersion)
            .order_by(GlobalBackboneVersion.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def queued_client_ids(self) -> list[str]:
        return list(self._queue.keys())

    @property
    def rounds_completed(self) -> int:
        return self._rounds_completed

    def metrics_snapshot(self) -> dict[str, Any]:
        """
        Lightweight snapshot of the in-memory aggregation state for debugging
        and development dashboards. Does not expose model weights.
        """
        queued_client_ids = list(self._queue.keys())
        queued_client_count = len(queued_client_ids)

        if self._round_start is None or not queued_client_ids:
            round_elapsed = None
            seconds_until_timeout = None
        else:
            round_elapsed = time.monotonic() - self._round_start
            seconds_until_timeout = max(0.0, float(ROUND_TIMEOUT_SECONDS) - round_elapsed)

        return {
            "queued_client_ids": queued_client_ids,
            "queued_client_count": queued_client_count,
            "rounds_completed": self._rounds_completed,
            "min_clients_per_round": MIN_CLIENTS_PER_ROUND,
            "round_timeout_seconds": ROUND_TIMEOUT_SECONDS,
            "round_elapsed_seconds": round_elapsed,
            "round_seconds_until_timeout": seconds_until_timeout,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _should_trigger(self) -> bool:
        return len(self._queue) >= MIN_CLIENTS_PER_ROUND

    def _majority_algorithm(self) -> str:
        """Pick the algorithm used by the majority of queued clients."""
        counts: dict[str, int] = {}
        for upload in self._queue.values():
            counts[upload.algorithm] = counts.get(upload.algorithm, 0) + 1
        return max(counts, key=lambda k: counts[k])

    async def _run_fedavg(self, algorithm: str, db: AsyncSession) -> None:
        """
        Execute FedAvg over all queued uploads and persist the result.

        w* = Σ (n_k / n_total) * w_k

        Only uploads matching the chosen algorithm are included.
        Any uploads for the other algorithm are carried forward.
        """
        eligible = {
            cid: u for cid, u in self._queue.items()
            if u.algorithm == algorithm
        }
        carry_forward = {
            cid: u for cid, u in self._queue.items()
            if u.algorithm != algorithm
        }

        if not eligible:
            logger.warning("FedAvg triggered but no eligible uploads for algorithm '%s'.", algorithm)
            return

        n_total = sum(u.interaction_count for u in eligible.values())

        # Weighted average for each parameter tensor
        aggregated: dict[str, np.ndarray] = {}
        for key in next(iter(eligible.values())).weights.keys():
            aggregated[key] = sum(
                (u.interaction_count / n_total) * u.weights[key]
                for u in eligible.values()
            )

        # Serialise: JSON → gzip → base64
        weights_json = {k: v.tolist() for k, v in aggregated.items()}
        compressed = gzip.compress(json.dumps(weights_json).encode())
        blob = base64.b64encode(compressed).decode()

        # Persist to Postgres
        new_version = GlobalBackboneVersion(
            weights_blob=blob,
            algorithm=algorithm,
            client_count=len(eligible),
            total_interactions=n_total,
        )
        db.add(new_version)
        await db.flush()  # populate auto-increment version before logging
        await db.commit()

        self._rounds_completed += 1
        logger.info(
            "FedAvg round %d complete — version=%d algorithm=%s clients=%d interactions=%d",
            self._rounds_completed,
            new_version.version,
            algorithm,
            len(eligible),
            n_total,
        )

        # Clear eligible uploads; carry forward any others
        self._queue = carry_forward
        self._round_start = time.monotonic() if carry_forward else None


# ---------------------------------------------------------------------------
# Serialisation helpers used by the download endpoint
# ---------------------------------------------------------------------------
def decode_backbone_blob(blob: str) -> dict[str, list]:
    """Decompress and deserialise a backbone blob from Postgres."""
    compressed = base64.b64decode(blob.encode())
    weights_json = gzip.decompress(compressed).decode()
    return json.loads(weights_json)