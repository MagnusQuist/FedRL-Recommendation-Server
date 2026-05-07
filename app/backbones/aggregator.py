"""
FL Aggregation Service
======================
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.db.models.aggregation_events import AggregationEvent
from app.db.models.federated import FederatedModel
from app.logger import logger

# ---------------------------------------------------------------------------
# Configuration — overridable via environment variables
# ---------------------------------------------------------------------------
# K — buffer size; once this many client uploads are queued, a FedBuff round runs.
CLIENTS_PER_ROUND = int(os.getenv("FEDERATED_CLIENTS_PER_ROUND", "3"))

# η_g — server learning rate applied to the aggregated delta.
# w^{t+1} = w^t − η_g · Δ̄^t.  1.0 reproduces "apply the averaged delta as-is".
SERVER_LR = float(os.getenv("FEDERATED_SERVER_LR", "1.0"))

# α — staleness scaling exponent. s(τ) = 1 / (1 + τ)^α.
# 0.5 matches the FedBuff paper; set to 0 to disable staleness scaling.
STALENESS_ALPHA = float(os.getenv("FEDERATED_STALENESS_ALPHA", "0.5"))


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
# Pure CPU-bound helpers (safe to run in a worker thread)
# ---------------------------------------------------------------------------
def _fedbuff_and_serialize(
    current_weights: dict[str, np.ndarray],
    eligible: list[QueuedUpload],
    base_weights_by_version: dict[int, dict[str, np.ndarray]],
    current_version: int,
    server_lr: float,
    staleness_alpha: float,
) -> str:
    """
    Buffered asynchronous aggregation (FedBuff) with hybrid n_k × staleness
    weighting, returning the gzip+base64 blob for ``FederatedModel.weights_blob``.

    For each client i in the buffer:
        Δ_i = w_base_i − w_local_i               (server-computed delta)
        τ_i = max(0, current_version − backbone_version_i)
        s_i = (1 / (1 + τ_i)^α) · n_i            (hybrid weight)

    Aggregate:  Δ̄ = Σ s_i · Δ_i / Σ s_i
    Apply:      w^{t+1} = w^t − η_g · Δ̄
    """
    param_keys = list(current_weights.keys())

    weighted_deltas: list[tuple[float, dict[str, np.ndarray]]] = []
    weight_sum = 0.0
    for u in eligible:
        base = base_weights_by_version[u.backbone_version]
        tau = max(0, current_version - u.backbone_version)
        scale = (1.0 / (1.0 + tau) ** staleness_alpha) * float(u.interaction_count)
        delta = {k: base[k] - u.weights[k] for k in param_keys}
        weighted_deltas.append((scale, delta))
        weight_sum += scale

    if weight_sum <= 0:
        raise ValueError("FedBuff weight normalizer is non-positive.")

    avg_delta: dict[str, np.ndarray] = {}
    for key in param_keys:
        accum = np.zeros_like(current_weights[key])
        for scale, delta in weighted_deltas:
            accum += scale * delta[key]
        avg_delta[key] = accum / weight_sum

    new_weights = {
        k: current_weights[k] - server_lr * avg_delta[k] for k in param_keys
    }

    weights_json = {k: v.tolist() for k, v in new_weights.items()}
    compressed = gzip.compress(json.dumps(weights_json).encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


def _decode_blob_to_arrays(blob: str) -> dict[str, np.ndarray]:
    """Decode a persisted weights blob into a dict of numpy arrays."""
    decoded = decode_backbone_blob(blob)
    return {k: np.array(v, dtype=np.float32) for k, v in decoded.items()}


async def _load_versions_weights(
    db: AsyncSession, versions: set[int]
) -> dict[int, dict[str, np.ndarray]]:
    """
    Fetch and decode persisted weights for the given set of model versions.
    Versions absent from the DB are simply omitted from the returned dict.
    """
    if not versions:
        return {}
    result = await db.execute(
        select(FederatedModel).where(FederatedModel.version.in_(versions))
    )
    return {row.version: _decode_blob_to_arrays(row.weights_blob) for row in result.scalars()}


# ---------------------------------------------------------------------------
# Aggregator (singleton, held in app state)
# ---------------------------------------------------------------------------
class FLAggregator:
    def __init__(self) -> None:
        self._queue: dict[str, QueuedUpload] = {}
        self._rounds_completed: int = 0
        # Cached latest persisted version. Updated on startup via
        # ``try_load_persisted_state`` and after every successful FedBuff round.
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
                    await db.execute(select(func.max(FederatedModel.version)))
                ).scalar()

                completed_rounds = (
                    await db.execute(
                        select(func.count(AggregationEvent.aggregation_event_id))
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
        db: AsyncSession,
    ) -> tuple[bool, int]:
        """
        Add or replace a client's upload in the queue.

        Returns ``(round_triggered, queued_client_count)``. If a round triggers,
        FedBuff is run and the result persisted to Postgres.
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
                await self._run_fedbuff(db)

            return triggered, len(self._queue)

    async def get_current_version(
        self,
        db: AsyncSession,
    ) -> Optional[FederatedModel]:
        """Return the latest persisted federated backbone."""
        result = await db.execute(
            select(FederatedModel)
            .order_by(FederatedModel.version.desc())
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

    async def _run_fedbuff(self, db: AsyncSession) -> None:
        """
        Buffered asynchronous aggregation (FedBuff) with hybrid n_k × staleness
        weighting:

            Δ_i = w_base_i − w_local_i
            s_i = (1 / (1 + τ_i)^α) · n_i
            Δ̄  = Σ s_i · Δ_i / Σ s_i
            w^{t+1} = w^t − η_g · Δ̄
        """
        if not self._queue:
            logger.warning("FedBuff triggered but queue is empty.")
            return

        eligible = list(self._queue.values())

        # Server-side delta computation requires the current global model and the
        # base-version weights each client trained against.
        current_model = await self.get_current_version(db)
        if current_model is None:
            logger.error("FedBuff aborted — no global model exists yet.")
            return

        current_version = current_model.version
        current_weights = _decode_blob_to_arrays(current_model.weights_blob)

        base_versions_needed = {u.backbone_version for u in eligible}
        base_weights_by_version = await _load_versions_weights(db, base_versions_needed)

        missing_versions = base_versions_needed - base_weights_by_version.keys()
        if missing_versions:
            logger.warning(
                "FedBuff dropping uploads with missing base versions: %s",
                sorted(missing_versions),
            )
            eligible = [u for u in eligible if u.backbone_version in base_weights_by_version]

        if not eligible:
            logger.error("FedBuff aborted — no eligible uploads after base-version check.")
            return

        n_total = sum(u.interaction_count for u in eligible)
        if n_total <= 0:
            logger.warning("FedBuff aborted because total interactions is %d.", n_total)
            return

        # Staleness telemetry — useful for the experiment writeup.
        stalenesses = [
            max(0, current_version - u.backbone_version) for u in eligible
        ]
        mean_staleness = sum(stalenesses) / len(stalenesses)
        max_staleness = max(stalenesses)
        logger.info(
            "FedBuff staleness — current_version=%d mean=%.2f max=%d distribution=%s",
            current_version,
            mean_staleness,
            max_staleness,
            stalenesses,
        )

        base_versions = {u.backbone_version for u in eligible}

        # Offload the CPU-bound reduce + gzip + base64 to a worker thread to keep
        # the API event loop responsive when a round is triggered.
        aggregation_started_at = datetime.now(timezone.utc)
        aggregation_started_perf = time.perf_counter()
        blob = await asyncio.to_thread(
            _fedbuff_and_serialize,
            current_weights,
            eligible,
            base_weights_by_version,
            current_version,
            SERVER_LR,
            STALENESS_ALPHA,
        )
        aggregation_duration_ms = int(
            round((time.perf_counter() - aggregation_started_perf) * 1000)
        )

        model_version_before = (
            str(base_versions.pop())
            if len(base_versions) == 1
            else ",".join(str(version) for version in sorted(base_versions))
        )
        next_version = await self._next_version(db)

        new_backbone = FederatedModel(
            version=next_version,
            weights_blob=blob,
        )
        db.add(new_backbone)
        await db.flush()

        db.add(
            AggregationEvent(
                timestamp=aggregation_started_at,
                aggregation_duration_ms=aggregation_duration_ms,
                participating_clients_ids=[upload.client_id for upload in eligible],
                num_clients_in_round=len(eligible),
                total_interactions=n_total,
                model_version_before=model_version_before,
                model_version_after=str(next_version),
                model_size_bytes=len(blob.encode("utf-8")),
                logged_at=datetime.now(timezone.utc),
            )
        )

        logger.info(
            "Aggregation event logged: version=%d aggregation_duration_ms=%d "
            "num_clients_in_round=%d total_interactions=%d "
            "model_version_before=%s model_version_after=%s model_size_bytes=%d",
            next_version,
            aggregation_duration_ms,
            len(eligible),
            n_total,
            model_version_before,
            str(next_version),
            len(blob.encode("utf-8")),
        )

        await db.commit()
        await db.refresh(new_backbone)

        self.model_version = new_backbone.version
        self._rounds_completed += 1
        logger.info(
            "FedBuff round complete — version=%d clients=%d interactions=%d "
            "mean_staleness=%.2f max_staleness=%d server_lr=%.3f staleness_alpha=%.3f "
            "aggregation_duration_ms=%d rounds_completed=%d",
            new_backbone.version,
            len(eligible),
            n_total,
            mean_staleness,
            max_staleness,
            SERVER_LR,
            STALENESS_ALPHA,
            aggregation_duration_ms,
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
