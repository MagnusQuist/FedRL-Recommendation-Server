"""Centralized training service orchestration."""

from __future__ import annotations

import asyncio
import os
import time
import tracemalloc
from datetime import datetime, timezone
from typing import Any

import torch
from sqlalchemy import func, select

from app.db.models import CentralizedModel, CentralizedTrainingEvent
from app.db.session import AsyncSessionLocal
from app.logging import logger
from app.ml.centralized.codec import (
    decode_json_blob,
    decode_tuples,
    encode_json_blob,
    module_state_to_json,
)
from app.ml.centralized.heads import (
    TSItemHead,
    TSNudgeHead,
    TSPriceHead,
    apply_tuple_to_heads,
)
from app.ml.centralized.models import BackboneEncoder, RewardPredictor, build_optimizer
from app.ml.centralized.trainer import (
    diff_l2_norm_state_dict,
    evaluate_backbone_loss,
    flat_l2_norm_state_dict,
    retrain_backbone,
    reward_stats,
)

CLIENTS_PER_ROUND = int(os.getenv("CENTRALIZED_CLIENTS_PER_ROUND", "2"))
MAX_TUPLE_POOL_SIZE = int(os.getenv("MAX_TUPLE_POOL_SIZE", "2000"))


class CentralizedService:
    """FastAPI singleton for centralized queueing, training rounds, and persistence."""

    def __init__(self):
        self._lock = asyncio.Lock()

        self.backbone = BackboneEncoder()
        self.reward_predictor = RewardPredictor()
        self.backbone.eval()
        self.reward_predictor.eval()
        self._optimizer = build_optimizer(self.backbone, self.reward_predictor)

        self.item_head = TSItemHead()
        self.price_head = TSPriceHead()
        self.nudge_head = TSNudgeHead()

        self._tuple_pool: list[dict] = []
        self._pending_uploads: dict[str, list[dict]] = {}
        self.model_version: int = 0
        self._rounds_completed: int = 0

    async def try_load_persisted_state(self) -> bool:
        try:
            async with AsyncSessionLocal() as db:
                row = (
                    await db.execute(
                        select(CentralizedModel)
                        .order_by(CentralizedModel.version.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                completed_rounds = (
                    await db.execute(
                        select(func.count(CentralizedTrainingEvent.centralized_training_event_id))
                    )
                ).scalar() or 0

            if row is None:
                return False

            self._load_backbone(row.backbone_blob)
            self._load_reward_predictor(row.reward_predictor_blob)
            self.item_head.load_state_dict(decode_json_blob(row.item_head_blob))
            self.price_head.load_state_dict(decode_json_blob(row.price_head_blob))
            self.nudge_head.load_state_dict(decode_json_blob(row.nudge_head_blob))
            self._tuple_pool = decode_json_blob(row.tuple_pool_blob)
            self.model_version = row.version
            self._rounds_completed = int(completed_rounds)

            logger.info(
                "Centralized state restored from DB: version=%d, tuples=%d, rounds_completed=%d",
                self.model_version,
                len(self._tuple_pool),
                self._rounds_completed,
            )
            return True

        except Exception:
            logger.exception("Failed to load persisted centralized state; will re-initialise.")
            return False

    async def process_interactions(
        self,
        client_id: str,
        count: int,
        data: str,
    ) -> tuple[int, bool, int]:
        """Buffer one client's tuples. Exactly CLIENTS_PER_ROUND unique clients trigger a round."""
        tuples = decode_tuples(data)
        self._validate_tuples(tuples)

        async with self._lock:
            upload_version = self.model_version
            for interaction in tuples:
                interaction["model_version_at_upload"] = upload_version

            if client_id in self._pending_uploads:
                logger.warning(
                    "Centralized: client '%s' re-uploaded before round triggered; "
                    "replacing %d previously buffered tuples with %d new tuples.",
                    client_id,
                    len(self._pending_uploads[client_id]),
                    len(tuples),
                )
            self._pending_uploads[client_id] = tuples

            queued = len(self._pending_uploads)
            buffered_total = sum(len(upload) for upload in self._pending_uploads.values())
            logger.info(
                "Centralized: buffered %d tuples from '%s'; %d/%d clients ready "
                "(round_buffered_tuples=%d, stamp_version=%d, declared_count=%d)",
                len(tuples),
                client_id,
                queued,
                CLIENTS_PER_ROUND,
                buffered_total,
                upload_version,
                count,
            )

            triggered = queued == CLIENTS_PER_ROUND
            if triggered:
                await self._run_training_round()

            return self.model_version, triggered, len(self._pending_uploads)

    async def _run_training_round(self) -> None:
        batch_clients = len(self._pending_uploads)
        batch_tuples = [
            interaction
            for tuples in self._pending_uploads.values()
            for interaction in tuples
        ]

        if not batch_tuples:
            logger.warning(
                "Centralized training round triggered with empty tuple buffer (clients=%d); skipping.",
                batch_clients,
            )
            self._pending_uploads = {}
            return

        existing_pool = list(self._tuple_pool)
        self._tuple_pool.extend(batch_tuples)
        self._tuple_pool = self._tuple_pool[-MAX_TUPLE_POOL_SIZE:]

        loss_before = evaluate_backbone_loss(
            self.backbone,
            self.reward_predictor,
            self._tuple_pool,
        )
        self._log_round_diagnostics(existing_pool, batch_tuples)

        round_started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        started_cpu = time.process_time()
        tracemalloc.start()

        for interaction in batch_tuples:
            apply_tuple_to_heads(
                self.backbone,
                self.item_head,
                self.price_head,
                self.nudge_head,
                interaction,
            )

        # Snapshot backbone + reward_predictor weights before training (cloned, detached, on CPU)
        state_before = {
            k: v.clone().detach().cpu()
            for k, v in {
                **self.backbone.state_dict(),
                **self.reward_predictor.state_dict(),
            }.items()
        }

        train_loss_last_epoch = await asyncio.to_thread(
            retrain_backbone,
            self.backbone,
            self.reward_predictor,
            self._optimizer,
            self._tuple_pool,
        )

        # Collect post-training weights; detach to CPU for norm arithmetic
        state_after = {
            k: v.detach().cpu()
            for k, v in {
                **self.backbone.state_dict(),
                **self.reward_predictor.state_dict(),
            }.items()
        }
        # ||w_after - w_before||₂ — magnitude of the weight shift this round
        model_update_norm = diff_l2_norm_state_dict(state_before, state_after)
        # Relative shift: normalised by pre-training model scale
        pre_training_norm = flat_l2_norm_state_dict(state_before)
        model_relative_update_norm = (
            model_update_norm / pre_training_norm if pre_training_norm > 0.0 else 0.0
        )

        loss_after = evaluate_backbone_loss(
            self.backbone,
            self.reward_predictor,
            self._tuple_pool,
        )

        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_wall = time.perf_counter() - started_perf
        elapsed_cpu = time.process_time() - started_cpu
        training_duration_ms = int(round(elapsed_wall * 1000))
        loss_delta = float(loss_after - loss_before)
        # Positive when loss decreased (model improved)
        bce_loss_improvement = float(loss_before - loss_after)

        model_version_before = self.model_version
        self.model_version += 1
        await self._persist_to_db(
            client_count=batch_clients,
            num_interactions=len(batch_tuples),
            total_training_interactions=len(self._tuple_pool),
            contributing_client_ids=sorted(self._pending_uploads.keys()),
            training_duration_ms=training_duration_ms,
            model_version_before=model_version_before,
            cpu_usage_percentage=(
                (elapsed_cpu / elapsed_wall) * 100 if elapsed_wall > 0 else 0.0
            ),
            memory_usage_mb=peak_memory_bytes / (1024 * 1024),
            loss_before=loss_before,
            loss_after=loss_after,
            loss_delta=loss_delta,
            bce_loss_improvement=bce_loss_improvement,
            model_update_norm_l2=model_update_norm,
            model_relative_update_norm_l2=model_relative_update_norm,
            training_round=self._rounds_completed + 1,
            timestamp=round_started_at,
        )

        self._rounds_completed += 1
        logger.info(
            "Centralized training round complete; version=%d clients=%d "
            "round_tuples=%d pool_size=%d loss_before=%.6f loss_after=%.6f "
            "loss_delta=%+.6f train_loss_last_epoch=%.6f training_duration_ms=%d "
            "rounds_completed=%d",
            self.model_version,
            batch_clients,
            len(batch_tuples),
            len(self._tuple_pool),
            loss_before,
            loss_after,
            loss_delta,
            train_loss_last_epoch,
            training_duration_ms,
            self._rounds_completed,
        )
        self._pending_uploads = {}

    def _validate_tuples(self, tuples: list[dict]) -> None:
        expected_ctx_dim = self.backbone.backbone[0].in_features
        for index, interaction in enumerate(tuples):
            ctx_len = len(interaction.get("context", []))
            if ctx_len != expected_ctx_dim:
                raise ValueError(
                    f"Tuple {index} has context_dim={ctx_len}, expected {expected_ctx_dim}. "
                    "Client and server context vector versions are out of sync."
                )

    def _load_backbone(self, blob: str) -> None:
        state = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in decode_json_blob(blob).items()
        }
        self.backbone.load_state_dict(state)
        self.backbone.eval()

    def _load_reward_predictor(self, blob: str) -> None:
        state = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in decode_json_blob(blob).items()
        }
        self.reward_predictor.load_state_dict(state)
        self.reward_predictor.eval()

    def _log_round_diagnostics(
        self,
        existing_pool: list[dict],
        batch_tuples: list[dict],
    ) -> None:
        loss_pre_existing = evaluate_backbone_loss(
            self.backbone,
            self.reward_predictor,
            existing_pool,
        )
        loss_pre_new = evaluate_backbone_loss(
            self.backbone,
            self.reward_predictor,
            batch_tuples,
        )
        reward_existing = reward_stats(existing_pool)
        reward_new = reward_stats(batch_tuples)

        logger.info(
            "Centralized round diagnostics; pool_size=%d existing=%d new=%d "
            "loss_pre_existing=%.6f loss_pre_new=%.6f "
            "reward_existing(mean=%.3f std=%.3f range=[%.3f, %.3f]) "
            "reward_new(mean=%.3f std=%.3f range=[%.3f, %.3f])",
            len(self._tuple_pool),
            reward_existing["n"],
            reward_new["n"],
            loss_pre_existing,
            loss_pre_new,
            reward_existing["mean"],
            reward_existing["std"],
            reward_existing["min"],
            reward_existing["max"],
            reward_new["mean"],
            reward_new["std"],
            reward_new["min"],
            reward_new["max"],
        )

    async def _persist_to_db(
        self,
        client_count: int,
        num_interactions: int,
        total_training_interactions: int,
        contributing_client_ids: list[str],
        training_duration_ms: int,
        model_version_before: int,
        cpu_usage_percentage: float,
        memory_usage_mb: float,
        loss_before: float,
        loss_after: float | None,
        loss_delta: float | None,
        bce_loss_improvement: float | None,
        model_update_norm_l2: float | None,
        model_relative_update_norm_l2: float | None,
        training_round: int | None,
        timestamp: datetime,
    ) -> None:
        blobs = self._model_blobs()

        async with AsyncSessionLocal() as db:
            row = CentralizedModel(version=self.model_version, **blobs)
            db.add(row)
            await db.flush()

            model_size_bytes = sum(
                len(blob.encode("utf-8")) for blob in blobs.values()
            )
            db.add(CentralizedTrainingEvent(
                timestamp=timestamp,
                training_duration_ms=training_duration_ms,
                num_interactions=num_interactions,
                num_clients_contributing=client_count,
                contributing_client_ids=contributing_client_ids,
                cpu_usage_percentage=cpu_usage_percentage,
                memory_usage_mb=memory_usage_mb,
                loss_before=loss_before,
                loss_after=loss_after,
                loss_delta=loss_delta,
                model_version_before=str(model_version_before),
                model_version_after=str(self.model_version),
                model_size_bytes=model_size_bytes,
                logged_at=datetime.now(timezone.utc),
                training_round=training_round,
                total_training_interactions=total_training_interactions,
                bce_loss_improvement=bce_loss_improvement,
                model_update_norm_l2=model_update_norm_l2,
                model_relative_update_norm_l2=model_relative_update_norm_l2,
            ))

            logger.info(
                "Centralized training event logged: num_clients_contributing=%d "
                "model_version_before=%s model_version_after=%s model_size_bytes=%d",
                client_count,
                model_version_before,
                self.model_version,
                model_size_bytes,
            )
            await db.commit()

    def _model_blobs(self) -> dict[str, str]:
        return {
            "backbone_blob": encode_json_blob(module_state_to_json(self.backbone)),
            "reward_predictor_blob": encode_json_blob(
                module_state_to_json(self.reward_predictor)
            ),
            "item_head_blob": encode_json_blob(self.item_head.state_dict()),
            "price_head_blob": encode_json_blob(self.price_head.state_dict()),
            "nudge_head_blob": encode_json_blob(self.nudge_head.state_dict()),
            "tuple_pool_blob": encode_json_blob(self._tuple_pool),
        }

    def get_model_snapshot(self) -> dict[str, Any]:
        blobs = self._model_blobs()
        return {
            "version": self.model_version,
            "backbone_weights": blobs["backbone_blob"],
            "reward_predictor_weights": blobs["reward_predictor_blob"],
            "head_weights": {
                "item": blobs["item_head_blob"],
                "price": blobs["price_head_blob"],
                "nudge": blobs["nudge_head_blob"],
            },
        }
