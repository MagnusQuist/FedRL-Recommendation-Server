"""FL aggregation service using FedBuff with n_k x staleness weighting."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AggregationEvent, FederatedModel
from app.db.session import AsyncSessionLocal
from app.logging import logger
from app.ml.federated.backbone_codec import (
    decode_backbone_arrays,
    decode_backbone_blob as decode_backbone_blob,
    encode_backbone_blob,
)
from app.ml.federated.fedbuff import aggregate_fedbuff


CLIENTS_PER_ROUND = int(os.getenv("FEDERATED_CLIENTS_PER_ROUND", "2"))
SERVER_LR = float(os.getenv("FEDERATED_SERVER_LR", "1.0"))
STALENESS_ALPHA = float(os.getenv("FEDERATED_STALENESS_ALPHA", "0.5"))


def _flat_l2_norm(weights: dict[str, np.ndarray]) -> float:
    """L2 norm of all parameter tensors concatenated into one vector."""
    return float(np.sqrt(sum(np.sum(v.astype(np.float64) ** 2) for v in weights.values())))


def _diff_l2_norm(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """L2 norm of (a[k] - b[k]) across all parameter tensors."""
    return float(np.sqrt(sum(np.sum((a[k].astype(np.float64) - b[k].astype(np.float64)) ** 2) for k in a)))


@dataclass
class QueuedUpload:
    client_id: str
    backbone_version: int
    interaction_count: int
    weights: dict[str, np.ndarray]
    received_at: float = field(default_factory=time.monotonic)


async def _load_versions_weights(
    db: AsyncSession, versions: set[int]
) -> dict[int, dict[str, np.ndarray]]:
    if not versions:
        return {}

    result = await db.execute(
        select(FederatedModel).where(FederatedModel.version.in_(versions))
    )
    return {
        row.version: decode_backbone_arrays(row.weights_blob) for row in result.scalars()
    }


class FLAggregator:
    """In-memory FL queue and round orchestrator, held as a FastAPI singleton."""

    def __init__(self) -> None:
        self._queue: dict[str, QueuedUpload] = {}
        self._rounds_completed: int = 0
        self.model_version: int = 0
        self._lock = asyncio.Lock()

    async def try_load_persisted_state(self) -> bool:
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
                "Failed to load persisted FL aggregator state; starting fresh."
            )
            return False

    async def enqueue(
        self,
        client_id: str,
        backbone_version: int,
        interaction_count: int,
        weights_dict: dict[str, list],
        db: AsyncSession,
    ) -> tuple[bool, int]:
        """Add or replace a client's upload."""
        weights = {
            key: np.array(value, dtype=np.float32) for key, value in weights_dict.items()
        }

        async with self._lock:
            self._queue[client_id] = QueuedUpload(
                client_id=client_id,
                backbone_version=backbone_version,
                interaction_count=interaction_count,
                weights=weights,
            )

            queued = len(self._queue)
            logger.info(
                "Queued upload from '%s'; %d/%d clients ready (n_k=%d)",
                client_id,
                queued,
                CLIENTS_PER_ROUND,
                interaction_count,
            )

            if queued > CLIENTS_PER_ROUND:
                logger.error(
                    "Federated queue size %d exceeds configured K=%d; draining all.",
                    queued,
                    CLIENTS_PER_ROUND,
                )

            triggered = queued == CLIENTS_PER_ROUND
            if triggered:
                await self._run_fedbuff(db)

            return triggered, len(self._queue)

    async def get_current_version(self, db: AsyncSession) -> FederatedModel | None:
        result = await db.execute(
            select(FederatedModel).order_by(FederatedModel.version.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    def queued_client_ids(self) -> list[str]:
        return list(self._queue.keys())

    def rounds_completed(self) -> int:
        return self._rounds_completed

    def metrics_snapshot(self) -> dict[str, Any]:
        queued_uploads = list(self._queue.values())
        queued_client_ids = [upload.client_id for upload in queued_uploads]
        queued_client_count = len(queued_client_ids)
        queued_total_interactions = sum(
            upload.interaction_count for upload in queued_uploads
        )
        queued_avg_interactions = (
            queued_total_interactions / queued_client_count
            if queued_client_count
            else None
        )
        oldest_received = min(
            (upload.received_at for upload in queued_uploads), default=None
        )
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

    async def _next_version(self, db: AsyncSession) -> int:
        latest = await self.get_current_version(db)
        return 1 if latest is None else latest.version + 1

    async def _run_fedbuff(self, db: AsyncSession) -> None:
        if not self._queue:
            logger.warning("FedBuff triggered but queue is empty.")
            return

        try:
            current_model = await self.get_current_version(db)
            if current_model is None:
                logger.error("FedBuff aborted; no global model exists yet.")
                return

            current_version = current_model.version
            current_weights = decode_backbone_arrays(current_model.weights_blob)

            eligible = list(self._queue.values())
            base_versions_needed = {upload.backbone_version for upload in eligible}
            base_weights_by_version = await _load_versions_weights(
                db, base_versions_needed
            )
            eligible = _uploads_with_base_weights(eligible, base_weights_by_version)

            round_stats = _round_stats(eligible, current_version)
            if round_stats is None:
                logger.error("FedBuff aborted; no eligible uploads after base-version check.")
                return

            if round_stats["total_interactions"] <= 0:
                logger.warning(
                    "FedBuff aborted because total interactions is %d.",
                    round_stats["total_interactions"],
                )
                return

            logger.info(
                "FedBuff staleness; current_version=%d mean=%.2f max=%d distribution=%s",
                current_version,
                round_stats["mean_staleness"],
                round_stats["max_staleness"],
                round_stats["stalenesses"],
            )

            aggregation_started_at = datetime.now(timezone.utc)
            start_perf = time.perf_counter()
            new_weights = await asyncio.to_thread(
                aggregate_fedbuff,
                current_weights,
                eligible,
                base_weights_by_version,
                current_version,
                SERVER_LR,
                STALENESS_ALPHA,
            )
            duration_ms = int(round((time.perf_counter() - start_perf) * 1000))

            backbone_update_norm = _diff_l2_norm(new_weights, current_weights)
            prev_norm = _flat_l2_norm(current_weights)
            backbone_relative_update_norm = backbone_update_norm / prev_norm if prev_norm > 0.0 else 0.0

            client_norms = [
                _diff_l2_norm(base_weights_by_version[upload.backbone_version], upload.weights)
                for upload in eligible
            ]
            mean_client_norm = float(np.mean(client_norms))
            std_client_norm = float(np.std(client_norms))

            blob = encode_backbone_blob(new_weights)

            next_version = await self._next_version(db)
            new_backbone = FederatedModel(version=next_version, weights_blob=blob)
            db.add(new_backbone)
            await db.flush()

            db.add(
                AggregationEvent(
                    timestamp=aggregation_started_at,
                    aggregation_duration_ms=duration_ms,
                    participating_clients_ids=[upload.client_id for upload in eligible],
                    num_clients_in_round=len(eligible),
                    total_interactions=round_stats["total_interactions"],
                    model_version_before=round_stats["base_versions"],
                    model_version_after=str(next_version),
                    model_size_bytes=len(blob.encode("utf-8")),
                    logged_at=datetime.now(timezone.utc),
                    aggregation_round=self._rounds_completed + 1,
                    previous_global_model_version=current_version,
                    backbone_update_norm_l2=backbone_update_norm,
                    backbone_relative_update_norm_l2=backbone_relative_update_norm,
                    mean_client_update_norm=mean_client_norm,
                    std_client_update_norm=std_client_norm,
                    aggregation_threshold_k=CLIENTS_PER_ROUND,
                )
            )

            logger.info(
                "FedBuff round complete; version=%d clients=%d interactions=%d "
                "mean_staleness=%.2f max_staleness=%d server_lr=%.3f "
                "staleness_alpha=%.3f aggregation_duration_ms=%d "
                "rounds_completed=%d",
                next_version,
                len(eligible),
                round_stats["total_interactions"],
                round_stats["mean_staleness"],
                round_stats["max_staleness"],
                SERVER_LR,
                STALENESS_ALPHA,
                duration_ms,
                self._rounds_completed + 1,
            )

            await db.commit()
            await db.refresh(new_backbone)

            self.model_version = new_backbone.version
            self._rounds_completed += 1

        finally:
            self._queue.clear()


def _uploads_with_base_weights(
    uploads: list[QueuedUpload],
    base_weights_by_version: dict[int, dict[str, np.ndarray]],
) -> list[QueuedUpload]:
    missing_versions = {
        upload.backbone_version
        for upload in uploads
        if upload.backbone_version not in base_weights_by_version
    }
    if missing_versions:
        logger.warning(
            "FedBuff dropping uploads with missing base versions: %s",
            sorted(missing_versions),
        )

    return [
        upload
        for upload in uploads
        if upload.backbone_version in base_weights_by_version
    ]


def _round_stats(
    uploads: list[QueuedUpload], current_version: int
) -> dict[str, Any] | None:
    if not uploads:
        return None

    stalenesses = [
        max(0, current_version - upload.backbone_version) for upload in uploads
    ]
    base_versions = sorted({upload.backbone_version for upload in uploads})
    return {
        "total_interactions": sum(upload.interaction_count for upload in uploads),
        "stalenesses": stalenesses,
        "mean_staleness": sum(stalenesses) / len(stalenesses),
        "max_staleness": max(stalenesses),
        "base_versions": ",".join(str(version) for version in base_versions),
    }
