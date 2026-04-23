"""
Centralized Training Service
=============================
Manages a centralized backbone + global heads for the centralized experiment arm.

Design decisions reflected here:
- Receives raw interaction tuples from centralized-mode clients.
- Uploads are buffered in memory, keyed by ``client_id``. Multiple uploads
  from the same client before a round triggers append to that client's
  tuple buffer but do not increase the unique-client count.
- A training round runs when exactly ``CENTRALIZED_CLIENTS_PER_ROUND``
  unique clients have uploaded — matching the FL aggregator's strict
  batch semantics so the federated and centralized arms train on the
  same number of client contributions per round (fair experimental
  comparison).
- On round trigger: merge the buffered tuples into the persistent pool
  (bounded by ``MAX_TUPLE_POOL_SIZE``), retrain the backbone on the full
  pool, apply Bayesian online updates to the global item/price/nudge
  heads for each tuple from the round, bump ``model_version``, and
  persist to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import random
import time
import tracemalloc
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import func, select

from app.db import AsyncSessionLocal
from app.db.models.centralized_training_events import CentralizedTrainingEvent
from app.db.models.centralized import CentralizedModel
from app.logger import logger

# ---------------------------------------------------------------------------
# Configuration — overridable via environment variables
# ---------------------------------------------------------------------------
CLIENTS_PER_ROUND = int(os.getenv("CENTRALIZED_CLIENTS_PER_ROUND", "2"))
MAX_TUPLE_POOL_SIZE = int(os.getenv("MAX_TUPLE_POOL_SIZE", "2000"))

RETRAIN_LR = 1e-3
RETRAIN_EPOCHS = 3
RETRAIN_BATCH_SIZE = 32
RETRAIN_WEIGHT_DECAY = 1e-4
RETRAIN_GRAD_CLIP = 1.0
CTX_PRICE_DELTA = 2  # index 2 in the 18-dim context vector

NUDGE_TYPES = ["N1", "N2", "N3", "N4", "N5", "N6"]


# ---------------------------------------------------------------------------
# Model architectures (exact copies of client-side models)
# ---------------------------------------------------------------------------
class BackboneEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, latent_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.backbone(x)

    def embed(self, context: list[float]) -> torch.Tensor:
        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.backbone(x).squeeze(0)


class RewardPredictor(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Head classes
# ---------------------------------------------------------------------------
class TSItemHead:
    def __init__(self, latent_dim=32, lam=1.0, v=0.5, max_items=200):
        self.latent_dim = latent_dim
        self.lam = lam
        self.v = v
        self.max_items = max_items
        self._params: OrderedDict[str, dict] = OrderedDict()

    def _init_item(self, item_id):
        d = self.latent_dim
        if len(self._params) >= self.max_items:
            self._params.popitem(last=False)
        entry = {"A": np.eye(d) * self.lam, "b": np.zeros(d)}
        self._params[item_id] = entry
        return entry

    def _touch(self, item_id):
        if item_id not in self._params:
            return self._init_item(item_id)
        self._params.move_to_end(item_id)
        return self._params[item_id]

    def update(self, item_id: str, embedding: np.ndarray, reward: float):
        p = self._touch(item_id)
        p["A"] += np.outer(embedding, embedding)
        p["b"] += reward * embedding

    def state_dict(self) -> dict:
        return {
            "params": {k: {"A": v["A"].tolist(), "b": v["b"].tolist()}
                       for k, v in self._params.items()},
            "lam": self.lam, "v": self.v, "max_items": self.max_items,
        }

    def load_state_dict(self, d: dict):
        self.lam = d.get("lam", self.lam)
        self.v = d.get("v", self.v)
        self.max_items = d.get("max_items", self.max_items)
        self._params = OrderedDict()
        for k, v in d.get("params", {}).items():
            self._params[k] = {"A": np.array(v["A"]), "b": np.array(v["b"])}


class TSPriceHead:
    def __init__(self, lam=1.0, v=0.5):
        self.lam = lam
        self.v = v
        self._A = lam
        self._b = 0.0

    def update(self, price_delta: float, reward: float):
        self._A += price_delta ** 2
        self._b += reward * price_delta

    def state_dict(self):
        return {"A": self._A, "b": self._b, "lam": self.lam, "v": self.v}

    def load_state_dict(self, d: dict):
        self._A = d["A"]
        self._b = d["b"]
        self.lam = d.get("lam", self.lam)
        self.v = d.get("v", self.v)


class TSNudgeHead:
    def __init__(self, prior_mu=0.5, prior_tau=1.0):
        self._params = {
            n: {"mu": prior_mu, "tau": prior_tau, "sum": 0.0, "count": 0}
            for n in NUDGE_TYPES
        }
        self._interaction_count = 0

    def update(self, nudge_type: str, reward: float):
        p = self._params[nudge_type]
        tau_obs = 1.0
        p["count"] += 1
        p["sum"] += reward
        p["tau"] += tau_obs
        p["mu"] = p["sum"] * tau_obs / p["tau"]
        self._interaction_count += 1

    def state_dict(self):
        return {"params": self._params, "interaction_count": self._interaction_count,
                "rr_index": 0, "last_reward": 0.0}

    def load_state_dict(self, d: dict):
        self._params = d["params"]
        self._interaction_count = d["interaction_count"]


# ---------------------------------------------------------------------------
# Encoding / decoding helpers
# ---------------------------------------------------------------------------
def _encode(obj: dict) -> str:
    """Serialize a dict to a gzip-compressed, base64-encoded JSON string."""
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def _decode(blob: str) -> dict:
    """Deserialize a blob produced by _encode."""
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode())


def decode_tuples(data: str) -> list[dict]:
    return json.loads(gzip.decompress(base64.b64decode(data)).decode())


def _backbone_to_serialisable(backbone: BackboneEncoder) -> dict:
    """Convert backbone state_dict to JSON-serialisable form."""
    return {k: v.tolist() for k, v in backbone.state_dict().items()}


# ---------------------------------------------------------------------------
# Backbone retraining
# ---------------------------------------------------------------------------
def retrain_backbone(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
    tuples: list[dict],
    seed: int | None = None,
) -> float:
    if not tuples:
        return 0.0

    device = next(backbone.parameters()).device
    rng = random.Random(seed)

    contexts = torch.tensor(
        [t["context"] for t in tuples], dtype=torch.float32, device=device
    )
    rewards = torch.tensor(
        [t["reward"] for t in tuples], dtype=torch.float32, device=device
    ).unsqueeze(1)

    n_samples = len(tuples)
    reward_mean = float(rewards.mean().item())
    reward_std = float(rewards.std(unbiased=False).item())
    reward_min = float(rewards.min().item())
    reward_max = float(rewards.max().item())

    # Pre-training diagnostic
    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        initial_embeddings = backbone(contexts)
        initial_predictions = reward_predictor(initial_embeddings)
        initial_loss = nn.functional.mse_loss(initial_predictions, rewards).item()
        initial_mae = float((initial_predictions - rewards).abs().mean().item())
        initial_emb_norm = float(initial_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain start: loss=%.4f mae=%.4f emb_norm=%.3f "
        "(%d tuples, reward_mean=%.3f, reward_std=%.3f, reward_range=[%.3f, %.3f])",
        initial_loss,
        initial_mae,
        initial_emb_norm,
        n_samples,
        reward_mean,
        reward_std,
        reward_min,
        reward_max,
    )

    backbone.train()
    reward_predictor.train()

    params = list(backbone.parameters()) + list(reward_predictor.parameters())
    optimizer = optim.Adam(
        params,
        lr=RETRAIN_LR,
        weight_decay=RETRAIN_WEIGHT_DECAY,
    )

    final_epoch_loss = 0.0
    final_epoch_mae = 0.0
    final_grad_norm_pre_clip = 0.0

    for epoch in range(RETRAIN_EPOCHS):
        indices = list(range(n_samples))
        rng.shuffle(indices)

        epoch_loss = 0.0
        epoch_abs_err_sum = 0.0
        epoch_seen = 0
        epoch_grad_norm_sum = 0.0
        n_batches = 0

        for start in range(0, n_samples, RETRAIN_BATCH_SIZE):
            batch_idx = indices[start : start + RETRAIN_BATCH_SIZE]
            x_batch = contexts[batch_idx]
            y_batch = rewards[batch_idx]

            embeddings = backbone(x_batch)
            predictions = reward_predictor(embeddings)
            loss = nn.functional.mse_loss(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()

            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                params, max_norm=RETRAIN_GRAD_CLIP
            )
            optimizer.step()

            with torch.no_grad():
                batch_abs_err = (predictions - y_batch).abs().sum().item()
                epoch_abs_err_sum += float(batch_abs_err)
                epoch_seen += y_batch.numel()

            epoch_loss += float(loss.item())
            epoch_grad_norm_sum += float(grad_norm_pre_clip)
            n_batches += 1

        final_epoch_loss = epoch_loss / max(n_batches, 1)
        final_epoch_mae = epoch_abs_err_sum / max(epoch_seen, 1)
        final_grad_norm_pre_clip = epoch_grad_norm_sum / max(n_batches, 1)

        logger.info(
            "centralized_retrain epoch %d/%d: loss=%.4f mae=%.4f grad_norm=%.3f "
            "(%d batches)",
            epoch + 1,
            RETRAIN_EPOCHS,
            final_epoch_loss,
            final_epoch_mae,
            final_grad_norm_pre_clip,
            n_batches,
        )

    # Post-training summary
    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        final_embeddings = backbone(contexts)
        final_emb_norm = float(final_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain done: loss %.4f -> %.4f, mae %.4f -> %.4f, "
        "emb_norm %.3f -> %.3f, avg_grad_norm=%.3f",
        initial_loss,
        final_epoch_loss,
        initial_mae,
        final_epoch_mae,
        initial_emb_norm,
        final_emb_norm,
        final_grad_norm_pre_clip,
    )

    return final_epoch_loss


def evaluate_backbone_loss(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
    tuples: list[dict],
) -> float:
    """Compute MSE loss over tuples without updating model parameters."""
    if not tuples:
        return 0.0

    contexts = torch.tensor([t["context"] for t in tuples], dtype=torch.float32)
    rewards = torch.tensor([t["reward"] for t in tuples], dtype=torch.float32).unsqueeze(1)

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        emb = backbone(contexts)
        pred = reward_predictor(emb)
        loss = nn.functional.mse_loss(pred, rewards)
    return float(loss.item())


# ---------------------------------------------------------------------------
# Head update — apply one tuple to the global heads
# ---------------------------------------------------------------------------
def apply_tuple_to_heads(backbone: BackboneEncoder, item_head: TSItemHead,
                         price_head: TSPriceHead, nudge_head: TSNudgeHead,
                         t: dict):
    ctx = t["context"]
    embedding = backbone.embed(ctx).numpy()
    price_delta = ctx[CTX_PRICE_DELTA] if len(ctx) > CTX_PRICE_DELTA else 0.0

    item_head.update(t["alternative_id"], embedding, t["reward"])
    price_head.update(price_delta, t["reward"])
    nudge_head.update(t["nudge_type"], t["reward"])


# ---------------------------------------------------------------------------
# CentralizedService — the main service object, held in app.state
# ---------------------------------------------------------------------------
class CentralizedService:
    """
    Manages the centralized backbone, reward predictor, global heads,
    tuple pool, and model versioning.  All state is persisted to the
    centralized_model_versions database table.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Models
        self.backbone = BackboneEncoder()
        self.reward_predictor = RewardPredictor()
        self.backbone.eval()
        self.reward_predictor.eval()

        # Global heads
        self.item_head = TSItemHead()
        self.price_head = TSPriceHead()
        self.nudge_head = TSNudgeHead()

        # Persistent tuple pool (bounded ring buffer via list + trim).
        # Survives restarts via centralized_model_versions.
        self._tuple_pool: list[dict] = []

        # Per-round client batch, in memory only (lost on restart, same as
        # the FL aggregator queue). A round triggers when exactly
        # ``CLIENTS_PER_ROUND`` unique clients have uploaded.
        self._pending_clients: set[str] = set()
        self._pending_tuples: list[dict] = []

        # Versioning
        self.model_version: int = 0
        self._rounds_completed: int = 0

    # ── Initialisation ─────────────────────────────────────────────────────

    async def try_load_persisted_state(self) -> bool:
        """
        Attempt to restore state from the database.
        Returns True if a persisted state was found and loaded.
        """
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(CentralizedModel)
                    .order_by(CentralizedModel.version.desc())
                    .limit(1)
                )
                row = result.scalar_one_or_none()

                completed_rounds = (
                    await db.execute(
                        select(func.count(CentralizedTrainingEvent.centralized_training_event_id))
                    )
                ).scalar() or 0

            if row is None:
                return False

            # Backbone
            backbone_data = _decode(row.backbone_blob)
            sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in backbone_data.items()}
            self.backbone.load_state_dict(sd)
            self.backbone.eval()

            # Reward predictor
            rp_data = _decode(row.reward_predictor_blob)
            rp_sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in rp_data.items()}
            self.reward_predictor.load_state_dict(rp_sd)
            self.reward_predictor.eval()

            # Heads
            self.item_head.load_state_dict(_decode(row.item_head_blob))
            self.price_head.load_state_dict(_decode(row.price_head_blob))
            self.nudge_head.load_state_dict(_decode(row.nudge_head_blob))

            # Tuple pool
            self._tuple_pool = _decode(row.tuple_pool_blob)

            self.model_version = row.version
            logger.info(
                "Centralized state restored from DB: version=%d, tuples=%d, rounds_completed=%d",
                self.model_version, len(self._tuple_pool), completed_rounds,
            )
            return True

        except Exception:
            logger.exception("Failed to load persisted centralized state — will re-initialise.")
            return False

    async def _persist_to_db(
        self,
        client_count: int,
        num_interactions: int,
        contributing_client_ids: list[str],
        training_duration_ms: int,
        model_version_before: int,
        cpu_usage_percentage: float,
        memory_usage_mb: float,
        loss_before: float,
        loss_after: float | None,
        loss_delta: float | None,
        timestamp: datetime,
    ) -> None:
        """Insert a new CentralizedModel row capturing the full current state."""
        backbone_blob = _encode(_backbone_to_serialisable(self.backbone))
        rp_blob = _encode({k: v.tolist() for k, v in self.reward_predictor.state_dict().items()})
        item_blob = _encode(self.item_head.state_dict())
        price_blob = _encode(self.price_head.state_dict())
        nudge_blob = _encode(self.nudge_head.state_dict())
        pool_blob = _encode(self._tuple_pool)

        async with AsyncSessionLocal() as db:
            row = CentralizedModel(
                version=self.model_version,
                backbone_blob=backbone_blob,
                reward_predictor_blob=rp_blob,
                item_head_blob=item_blob,
                price_head_blob=price_blob,
                nudge_head_blob=nudge_blob,
                tuple_pool_blob=pool_blob,
            )
            db.add(row)
            await db.flush()

            model_size_bytes = sum(
                len(blob.encode("utf-8"))
                for blob in (
                    backbone_blob,
                    rp_blob,
                    item_blob,
                    price_blob,
                    nudge_blob,
                    pool_blob,
                )
            )
            db.add(
                CentralizedTrainingEvent(
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
                )
            )

            logger.info(
                    "Centralized training event logged: num_clients_contributing=%d model_version_before=%s model_version_after=%s model_size_bytes=%d",
                    client_count,
                    model_version_before,
                    self.model_version,
                    model_size_bytes,
                )

            await db.commit()

    # ── Core processing ────────────────────────────────────────────────────

    async def process_interactions(
        self, client_id: str, count: int, data: str
    ) -> tuple[int, bool, int]:
        """
        Buffer a client's interaction tuples. When exactly
        ``CLIENTS_PER_ROUND`` unique clients have uploaded, run a training
        round (retrain backbone + update heads + persist + bump version).

        Returns ``(model_version, round_triggered, queued_client_count)``.
        ``model_version`` reflects the latest persisted model — it will only
        change when ``round_triggered`` is ``True``.
        """
        tuples = decode_tuples(data)

        async with self._lock:
            self._pending_clients.add(client_id)
            self._pending_tuples.extend(tuples)

            queued = len(self._pending_clients)
            logger.info(
                "Centralized: buffered %d tuples from '%s' — %d/%d clients ready "
                "(round_buffered_tuples=%d)",
                len(tuples),
                client_id,
                queued,
                CLIENTS_PER_ROUND,
                len(self._pending_tuples),
            )

            triggered = queued == CLIENTS_PER_ROUND
            if triggered:
                await self._run_training_round()

            return self.model_version, triggered, len(self._pending_clients)

    async def _run_training_round(self) -> None:
        """
        Merge the round's buffered tuples into the persistent pool, retrain
        the backbone on the full pool, apply head updates for the round's
        tuples, bump ``model_version``, persist, and clear the round buffer.

        Must be called with ``self._lock`` held.
        """
        batch_tuples = self._pending_tuples
        batch_clients = len(self._pending_clients)

        if not batch_tuples:
            logger.warning(
                "Centralized training round triggered with empty tuple buffer "
                "(clients=%d) — skipping retrain.",
                batch_clients,
            )
            self._pending_clients.clear()
            self._pending_tuples = []
            return

        self._tuple_pool.extend(batch_tuples)
        if len(self._tuple_pool) > MAX_TUPLE_POOL_SIZE:
            self._tuple_pool = self._tuple_pool[-MAX_TUPLE_POOL_SIZE:]

        loss_before = evaluate_backbone_loss(
            self.backbone,
            self.reward_predictor,
            self._tuple_pool,
        )

        # Start timing
        round_started_at = datetime.now(timezone.utc)
        round_started_perf = time.perf_counter()
        round_started_cpu = time.process_time()
        tracemalloc.start()

        # Retrain the backbone
        loss = await asyncio.to_thread(
            retrain_backbone, self.backbone, self.reward_predictor, self._tuple_pool
        )

        # Apply head updates
        for t in batch_tuples:
            apply_tuple_to_heads(
                self.backbone, self.item_head, self.price_head, self.nudge_head, t
            )

        # Get metrics
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_wall = time.perf_counter() - round_started_perf
        elapsed_cpu = time.process_time() - round_started_cpu
        training_duration_ms = int(
            round(elapsed_wall * 1000)
        )
        cpu_usage_percentage = (
            (elapsed_cpu / elapsed_wall) * 100 if elapsed_wall > 0 else 0.0
        )
        memory_usage_mb = peak_memory_bytes / (1024 * 1024)
        loss_delta = float(loss - loss_before)

        # Persist to database
        model_version_before = self.model_version
        self.model_version += 1
        await self._persist_to_db(
            client_count=batch_clients,
            num_interactions=len(batch_tuples),
            contributing_client_ids=sorted(self._pending_clients),
            training_duration_ms=training_duration_ms,
            model_version_before=model_version_before,
            cpu_usage_percentage=cpu_usage_percentage,
            memory_usage_mb=memory_usage_mb,
            loss_before=loss_before,
            loss_after=loss,
            loss_delta=loss_delta,
            timestamp=round_started_at,
        )

        # Log
        self._rounds_completed += 1
        logger.info(
            "Centralized training round complete — version=%d clients=%d "
            "round_tuples=%d pool_size=%d loss=%.6f training_duration_ms=%d "
            "rounds_completed=%d",
            self.model_version,
            batch_clients,
            len(batch_tuples),
            len(self._tuple_pool),
            loss,
            training_duration_ms,
            self._rounds_completed,
        )

        self._pending_clients.clear()
        self._pending_tuples = []

    # ── Model serialisation for GET endpoint ───────────────────────────────

    def get_model_snapshot(self) -> dict[str, Any]:
        """
        Build the response payload for GET /centralized/model.
        """
        backbone_dict = _backbone_to_serialisable(self.backbone)
        return {
            "version": self.model_version,
            "backbone_weights": _encode(backbone_dict),
            "head_weights": {
                "item": _encode(self.item_head.state_dict()),
                "price": _encode(self.price_head.state_dict()),
                "nudge": _encode(self.nudge_head.state_dict()),
            },
        }
