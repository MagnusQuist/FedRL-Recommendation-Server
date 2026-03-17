"""
FL Aggregation Service
======================
Manages in-memory upload queues and executes per-algorithm FedAvg rounds.

Design decisions reflected here:
- Uploads are queued in memory per algorithm and keyed by client_id.
- If the same client uploads twice before a round triggers for that algorithm,
  the newer upload replaces the older one.
- A round triggers independently per algorithm when:
    queued uploads >= min_clients_per_round
  OR
    round_timeout_seconds elapses since the first upload arrived for that algorithm.
- If round_timeout elapses but the queue is below min_clients_per_round,
  uploads are carried forward to the next round (not discarded).
- FedAvg: w* = Σ (n_k / n_total) * w_k — pure NumPy, no PyTorch required.
- The aggregated backbone is persisted to PostgreSQL with per-algorithm versioning.
"""

from __future__ import annotations

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
from app.db.seed_backbone import SUPPORTED_ALGORITHMS

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
    weights: dict[str, np.ndarray]
    received_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Aggregator (singleton, held in app state)
# ---------------------------------------------------------------------------
class FLAggregator:
    def __init__(self) -> None:
        self._queues: dict[str, dict[str, QueuedUpload]] = {
            algorithm: {} for algorithm in SUPPORTED_ALGORITHMS
        }
        self._round_starts: dict[str, Optional[float]] = {
            algorithm: None for algorithm in SUPPORTED_ALGORITHMS
        }
        self._rounds_completed: dict[str, int] = {
            algorithm: 0 for algorithm in SUPPORTED_ALGORITHMS
        }
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
        Add or replace a client's upload in the queue for the given algorithm.

        Returns (round_triggered, queued_client_count_for_algorithm).
        If a round triggers, FedAvg is run and the result persisted to Postgres.
        """
        self._validate_algorithm(algorithm)
        weights = {k: np.array(v, dtype=np.float32) for k, v in weights_dict.items()}

        async with self._lock:
            queue = self._queues[algorithm]
            queue[client_id] = QueuedUpload(
                client_id=client_id,
                backbone_version=backbone_version,
                interaction_count=interaction_count,
                algorithm=algorithm,
                weights=weights,
            )

            if self._round_starts[algorithm] is None:
                self._round_starts[algorithm] = time.monotonic()

            queued = len(queue)
            logger.info(
                "Queued upload from '%s' — %d/%d clients ready (algorithm=%s, n_k=%d)",
                client_id,
                queued,
                MIN_CLIENTS_PER_ROUND,
                algorithm,
                interaction_count,
            )

            triggered = self._should_trigger(algorithm)
            if triggered:
                await self._run_fedavg(algorithm, db)

            return triggered, len(self._queues[algorithm])

    async def check_timeout(self, db: AsyncSession) -> bool:
        """
        Called periodically by the background timeout task.

        Checks each algorithm queue independently.
        Returns True if at least one round was triggered.
        """
        triggered_any = False

        async with self._lock:
            for algorithm in SUPPORTED_ALGORITHMS:
                queue = self._queues[algorithm]
                round_start = self._round_starts[algorithm]

                if not queue or round_start is None:
                    continue

                elapsed = time.monotonic() - round_start
                if elapsed < ROUND_TIMEOUT_SECONDS:
                    continue

                queued = len(queue)
                if queued < MIN_CLIENTS_PER_ROUND:
                    logger.info(
                        "Round timeout elapsed for algorithm='%s' but only %d/%d clients queued — "
                        "carrying uploads forward.",
                        algorithm,
                        queued,
                        MIN_CLIENTS_PER_ROUND,
                    )
                    self._round_starts[algorithm] = time.monotonic()
                    continue

                await self._run_fedavg(algorithm, db)
                triggered_any = True

        return triggered_any

    async def get_current_version(
        self,
        db: AsyncSession,
        algorithm: str,
    ) -> Optional[GlobalBackboneVersion]:
        """Return the latest persisted backbone for the given algorithm."""
        self._validate_algorithm(algorithm)

        result = await db.execute(
            select(GlobalBackboneVersion)
            .where(GlobalBackboneVersion.algorithm == algorithm)
            .order_by(GlobalBackboneVersion.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def queued_client_ids(self, algorithm: str) -> list[str]:
        self._validate_algorithm(algorithm)
        return list(self._queues[algorithm].keys())

    def rounds_completed(self, algorithm: str) -> int:
        self._validate_algorithm(algorithm)
        return self._rounds_completed[algorithm]

    def metrics_snapshot(self) -> dict[str, Any]:
        """
        Lightweight snapshot of in-memory aggregation state for debugging.

        Does not expose model weights.
        """
        per_algorithm: dict[str, Any] = {}

        for algorithm in SUPPORTED_ALGORITHMS:
            queue = self._queues[algorithm]
            queued_uploads = list(queue.values())
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

            round_start = self._round_starts[algorithm]
            if round_start is None or not queued_client_ids:
                round_elapsed = None
                seconds_until_timeout = None
            else:
                round_elapsed = time.monotonic() - round_start
                seconds_until_timeout = max(0.0, float(ROUND_TIMEOUT_SECONDS) - round_elapsed)

            per_algorithm[algorithm] = {
                "queued_client_ids": queued_client_ids,
                "queued_client_count": queued_client_count,
                "queued_total_interactions": queued_total_interactions,
                "queued_avg_interactions": queued_avg_interactions,
                "queued_oldest_upload_age_seconds": oldest_upload_age_seconds,
                "rounds_completed": self._rounds_completed[algorithm],
                "round_elapsed_seconds": round_elapsed,
                "round_seconds_until_timeout": seconds_until_timeout,
            }

        return {
            "algorithms": per_algorithm,
            "min_clients_per_round": MIN_CLIENTS_PER_ROUND,
            "round_timeout_seconds": ROUND_TIMEOUT_SECONDS,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _validate_algorithm(self, algorithm: str) -> None:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Supported algorithms: {SUPPORTED_ALGORITHMS}"
            )

    def _should_trigger(self, algorithm: str) -> bool:
        return len(self._queues[algorithm]) >= MIN_CLIENTS_PER_ROUND

    async def _next_version(self, db: AsyncSession, algorithm: str) -> int:
        latest = await self.get_current_version(db, algorithm)
        return 1 if latest is None else latest.version + 1

    async def _run_fedavg(self, algorithm: str, db: AsyncSession) -> None:
        """
        Execute FedAvg over all queued uploads for one algorithm and persist the result.

        w* = Σ (n_k / n_total) * w_k
        """
        self._validate_algorithm(algorithm)

        queue = self._queues[algorithm]
        if not queue:
            logger.warning(
                "FedAvg triggered but queue is empty for algorithm '%s'.",
                algorithm,
            )
            return

        eligible = list(queue.values())
        n_total = sum(u.interaction_count for u in eligible)

        if n_total <= 0:
            logger.warning(
                "FedAvg aborted for algorithm='%s' because total interactions is %d.",
                algorithm,
                n_total,
            )
            return

        # Optional safety check: all uploads should target the same backbone version.
        base_versions = {u.backbone_version for u in eligible}
        if len(base_versions) > 1:
            logger.warning(
                "FedAvg for algorithm='%s' includes mixed base versions: %s",
                algorithm,
                sorted(base_versions),
            )

        aggregated: dict[str, np.ndarray] = {}
        param_keys = next(iter(eligible[0].weights.keys() for _ in [0]))

        for key in param_keys:
            aggregated[key] = sum(
                (u.interaction_count / n_total) * u.weights[key]
                for u in eligible
            )

        weights_json = {k: v.tolist() for k, v in aggregated.items()}
        compressed = gzip.compress(json.dumps(weights_json).encode("utf-8"))
        blob = base64.b64encode(compressed).decode("utf-8")

        next_version = await self._next_version(db, algorithm)

        new_backbone = GlobalBackboneVersion(
            version=next_version,
            weights_blob=blob,
            algorithm=algorithm,
            client_count=len(eligible),
            total_interactions=n_total,
        )
        db.add(new_backbone)
        await db.flush()
        await db.commit()
        await db.refresh(new_backbone)

        self._rounds_completed[algorithm] += 1
        logger.info(
            "FedAvg round complete — algorithm=%s version=%d clients=%d interactions=%d rounds_completed=%d",
            algorithm,
            new_backbone.version,
            len(eligible),
            n_total,
            self._rounds_completed[algorithm],
        )

        self._queues[algorithm].clear()
        self._round_starts[algorithm] = None