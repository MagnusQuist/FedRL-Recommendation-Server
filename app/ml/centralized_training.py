"""
Centralized Training Service
=============================
Manages a centralized backbone + global heads for the centralized experiment arm.
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

from app.db.session import AsyncSessionLocal
from app.db.models import CentralizedModel, CentralizedTrainingEvent
from app.logging import logger

# ---------------------------------------------------------------------------
# Configuration — overridable via environment variables
# ---------------------------------------------------------------------------
CLIENTS_PER_ROUND = int(os.getenv("CENTRALIZED_CLIENTS_PER_ROUND", "2"))
MAX_TUPLE_POOL_SIZE = int(os.getenv("MAX_TUPLE_POOL_SIZE", "2000"))

RETRAIN_LR_BACKBONE = 1e-4
RETRAIN_LR_PREDICTOR = 1e-3
RETRAIN_EPOCHS = 5
RETRAIN_BATCH_SIZE = 64
RETRAIN_WEIGHT_DECAY = 0.0
RETRAIN_GRAD_CLIP = 1.0
CTX_PRICE_DELTA = 2

NUDGE_TYPES = ["N1", "N2", "N3", "N4", "N5", "N6"]


# ---------------------------------------------------------------------------
# Model architectures (exact copies of client-side models)
# ---------------------------------------------------------------------------
class BackboneEncoder(nn.Module):
    def __init__(self, input_dim: int = 21, latent_dim: int = 32):
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
        self.net = nn.Linear(input_dim, 1)

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
            "params": {k: {"A": v["A"].tolist(), "b": v["b"].tolist()} for k, v in self._params.items()},
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
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def _decode(blob: str) -> dict:
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode())


def decode_tuples(data: str) -> list[dict]:
    return json.loads(gzip.decompress(base64.b64decode(data)).decode())


def _backbone_to_serialisable(backbone: BackboneEncoder) -> dict:
    return {k: v.tolist() for k, v in backbone.state_dict().items()}


# ---------------------------------------------------------------------------
# Backbone retraining
# ---------------------------------------------------------------------------
def _split_decay_params(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    return decay, no_decay


def retrain_backbone(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
    optimizer: optim.Optimizer,
    tuples: list[dict],
    seed: int | None = None,
) -> float:
    """
    Train backbone + reward_predictor jointly on tuples using binary cross-entropy
    (BCE) on accept/dismiss labels (reward > 0). Backbone Adam state is persistent
    across rounds.
    """
    if not tuples:
        return 0.0

    device = next(backbone.parameters()).device
    rng = random.Random(seed)

    contexts = torch.tensor([t["context"] for t in tuples], dtype=torch.float32, device=device)
    labels = (
        torch.tensor([t["reward"] for t in tuples], dtype=torch.float32, device=device) > 0
    ).float().unsqueeze(1)

    n_samples = len(tuples)

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        initial_embeddings = backbone(contexts)
        initial_logits = reward_predictor(initial_embeddings)
        initial_loss = nn.functional.binary_cross_entropy_with_logits(initial_logits, labels).item()
        initial_acc = float(((initial_logits > 0) == labels.bool()).float().mean().item())
        initial_emb_norm = float(initial_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain start: bce=%.4f acc=%.3f emb_norm=%.3f (%d tuples)",
        initial_loss, initial_acc, initial_emb_norm, n_samples,
    )

    backbone.train()
    reward_predictor.train()
    params = [p for group in optimizer.param_groups for p in group["params"]]

    final_epoch_loss = 0.0
    final_epoch_acc = 0.0
    final_grad_norm_pre_clip = 0.0

    for epoch in range(RETRAIN_EPOCHS):
        indices = list(range(n_samples))
        rng.shuffle(indices)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0
        epoch_grad_norm_sum = 0.0
        n_batches = 0

        for start in range(0, n_samples, RETRAIN_BATCH_SIZE):
            batch_idx = indices[start: start + RETRAIN_BATCH_SIZE]
            x_batch = contexts[batch_idx]
            labels_batch = labels[batch_idx]

            embeddings = backbone(x_batch)
            logits = reward_predictor(embeddings)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(params, max_norm=RETRAIN_GRAD_CLIP)
            optimizer.step()

            with torch.no_grad():
                epoch_correct += int(((logits > 0) == labels_batch.bool()).sum().item())
                epoch_seen += labels_batch.numel()
                epoch_loss += float(loss.item()) * labels_batch.numel()

            epoch_grad_norm_sum += float(grad_norm_pre_clip)
            n_batches += 1

        final_epoch_loss = epoch_loss / max(epoch_seen, 1)
        final_epoch_acc = epoch_correct / max(epoch_seen, 1)
        final_grad_norm_pre_clip = epoch_grad_norm_sum / max(n_batches, 1)

        logger.info(
            "centralized_retrain epoch %d/%d: bce=%.4f acc=%.3f grad_norm=%.3f (%d batches)",
            epoch + 1, RETRAIN_EPOCHS, final_epoch_loss, final_epoch_acc,
            final_grad_norm_pre_clip, n_batches,
        )

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        final_embeddings = backbone(contexts)
        final_emb_norm = float(final_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain done: bce %.4f -> %.4f, acc %.3f -> %.3f, "
        "emb_norm %.3f -> %.3f, avg_grad_norm=%.3f",
        initial_loss, final_epoch_loss, initial_acc, final_epoch_acc,
        initial_emb_norm, final_emb_norm, final_grad_norm_pre_clip,
    )

    return final_epoch_loss


def evaluate_backbone_loss(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
    tuples: list[dict],
) -> float:
    if not tuples:
        return 0.0

    contexts = torch.tensor([t["context"] for t in tuples], dtype=torch.float32)
    labels = (
        torch.tensor([t["reward"] for t in tuples], dtype=torch.float32) > 0
    ).float().unsqueeze(1)

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        emb = backbone(contexts)
        logits = reward_predictor(emb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return float(loss.item())


def _reward_stats(tuples: list[dict]) -> dict[str, float]:
    if not tuples:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.asarray([t["reward"] for t in tuples], dtype=np.float32)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size),
    }


def apply_tuple_to_heads(
    backbone: BackboneEncoder,
    item_head: TSItemHead,
    price_head: TSPriceHead,
    nudge_head: TSNudgeHead,
    t: dict,
):
    ctx = t["context"]
    embedding = backbone.embed(ctx).numpy()
    price_delta = ctx[CTX_PRICE_DELTA] if len(ctx) > CTX_PRICE_DELTA else 0.0
    item_head.update(t["alternative_id"], embedding, t["reward"])
    price_head.update(price_delta, t["reward"])
    nudge_head.update(t["nudge_type"], t["reward"])


# ---------------------------------------------------------------------------
# CentralizedService — singleton held in app.state
# ---------------------------------------------------------------------------
class CentralizedService:
    """
    Manages the centralized backbone, reward predictor, global heads,
    tuple pool, and model versioning. All state is persisted to the
    centralized_model_versions database table.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        self.backbone = BackboneEncoder()
        self.reward_predictor = RewardPredictor()
        self.backbone.eval()
        self.reward_predictor.eval()

        # Persistent Adam optimizer — m/v buffers carry across rounds.
        # Weight decay on weights only; never on biases.
        bb_decay, bb_no_decay = _split_decay_params(self.backbone)
        pred_decay, pred_no_decay = _split_decay_params(self.reward_predictor)
        self._optimizer = optim.Adam([
            {"params": bb_decay,      "lr": RETRAIN_LR_BACKBONE,  "weight_decay": RETRAIN_WEIGHT_DECAY},
            {"params": bb_no_decay,   "lr": RETRAIN_LR_BACKBONE,  "weight_decay": 0.0},
            {"params": pred_decay,    "lr": RETRAIN_LR_PREDICTOR, "weight_decay": RETRAIN_WEIGHT_DECAY},
            {"params": pred_no_decay, "lr": RETRAIN_LR_PREDICTOR, "weight_decay": 0.0},
        ])

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
                result = await db.execute(
                    select(CentralizedModel).order_by(CentralizedModel.version.desc()).limit(1)
                )
                row = result.scalar_one_or_none()

                completed_rounds = (
                    await db.execute(
                        select(func.count(CentralizedTrainingEvent.centralized_training_event_id))
                    )
                ).scalar() or 0

            if row is None:
                return False

            backbone_data = _decode(row.backbone_blob)
            sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in backbone_data.items()}
            self.backbone.load_state_dict(sd)
            self.backbone.eval()

            # Remap legacy keys (net.0.* → net.*) produced when net was nn.Sequential.
            rp_data = _decode(row.reward_predictor_blob)
            rp_sd = {
                k.replace("net.0.", "net.", 1): torch.tensor(v, dtype=torch.float32)
                for k, v in rp_data.items()
            }
            self.reward_predictor.load_state_dict(rp_sd)
            self.reward_predictor.eval()

            # Detect legacy zero-init predictor and replace with Kaiming-uniform.
            with torch.no_grad():
                weight_norm_sq = sum(
                    float(p.pow(2).sum().item())
                    for n, p in self.reward_predictor.named_parameters()
                    if n.endswith("weight")
                )
                if weight_norm_sq < 1e-12:
                    for n, p in self.reward_predictor.named_parameters():
                        if n.endswith("weight"):
                            bound = float(p.shape[-1]) ** -0.5
                            p.uniform_(-bound, bound)
                    logger.warning(
                        "Legacy zero-initialised reward predictor detected on load (version=%d). "
                        "Reinitialised with Kaiming-uniform.",
                        row.version,
                    )

            self.item_head.load_state_dict(_decode(row.item_head_blob))
            self.price_head.load_state_dict(_decode(row.price_head_blob))
            self.nudge_head.load_state_dict(_decode(row.nudge_head_blob))
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
                for blob in (backbone_blob, rp_blob, item_blob, price_blob, nudge_blob, pool_blob)
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
            ))

            logger.info(
                "Centralized training event logged: num_clients_contributing=%d "
                "model_version_before=%s model_version_after=%s model_size_bytes=%d",
                client_count, model_version_before, self.model_version, model_size_bytes,
            )
            await db.commit()

    async def process_interactions(self, client_id: str, count: int, data: str) -> tuple[int, bool, int]:
        """
        Buffer a client's interaction tuples. When exactly CLIENTS_PER_ROUND unique
        clients have uploaded, run a training round.

        Returns (model_version, round_triggered, queued_client_count).
        """
        tuples = decode_tuples(data)

        expected_ctx_dim = self.backbone.backbone[0].in_features
        for i, t in enumerate(tuples):
            ctx_len = len(t.get("context", []))
            if ctx_len != expected_ctx_dim:
                raise ValueError(
                    f"Tuple {i} has context_dim={ctx_len}, expected {expected_ctx_dim}. "
                    "Client and server context vector versions are out of sync."
                )

        async with self._lock:
            stamp_version = self.model_version
            for t in tuples:
                t["model_version_at_upload"] = stamp_version

            if client_id in self._pending_uploads:
                logger.warning(
                    "Centralized: client '%s' re-uploaded before round triggered — "
                    "replacing %d previously buffered tuples with %d new tuples.",
                    client_id, len(self._pending_uploads[client_id]), len(tuples),
                )
            self._pending_uploads[client_id] = tuples

            queued = len(self._pending_uploads)
            buffered_total = sum(len(v) for v in self._pending_uploads.values())
            logger.info(
                "Centralized: buffered %d tuples from '%s' — %d/%d clients ready "
                "(round_buffered_tuples=%d, stamp_version=%d)",
                len(tuples), client_id, queued, CLIENTS_PER_ROUND,
                buffered_total, stamp_version,
            )

            triggered = queued == CLIENTS_PER_ROUND
            if triggered:
                await self._run_training_round()

            return self.model_version, triggered, len(self._pending_uploads)

    async def _run_training_round(self) -> None:
        batch_clients = len(self._pending_uploads)
        batch_tuples: list[dict] = []
        for tuples in self._pending_uploads.values():
            batch_tuples.extend(tuples)

        if not batch_tuples:
            logger.warning(
                "Centralized training round triggered with empty tuple buffer (clients=%d) — skipping.",
                batch_clients,
            )
            self._pending_uploads = {}
            return

        existing_pool = list(self._tuple_pool)
        self._tuple_pool.extend(batch_tuples)
        if len(self._tuple_pool) > MAX_TUPLE_POOL_SIZE:
            self._tuple_pool = self._tuple_pool[-MAX_TUPLE_POOL_SIZE:]

        loss_before = evaluate_backbone_loss(self.backbone, self.reward_predictor, self._tuple_pool)

        loss_pre_existing = evaluate_backbone_loss(self.backbone, self.reward_predictor, existing_pool)
        loss_pre_new = evaluate_backbone_loss(self.backbone, self.reward_predictor, batch_tuples)
        reward_existing = _reward_stats(existing_pool)
        reward_new = _reward_stats(batch_tuples)

        logger.info(
            "Centralized round diagnostics — pool_size=%d existing=%d new=%d "
            "loss_pre_existing=%.6f loss_pre_new=%.6f "
            "reward_existing(mean=%.3f std=%.3f range=[%.3f, %.3f]) "
            "reward_new(mean=%.3f std=%.3f range=[%.3f, %.3f])",
            len(self._tuple_pool), reward_existing["n"], reward_new["n"],
            loss_pre_existing, loss_pre_new,
            reward_existing["mean"], reward_existing["std"],
            reward_existing["min"], reward_existing["max"],
            reward_new["mean"], reward_new["std"],
            reward_new["min"], reward_new["max"],
        )

        round_started_at = datetime.now(timezone.utc)
        round_started_perf = time.perf_counter()
        round_started_cpu = time.process_time()
        tracemalloc.start()

        # Apply head updates BEFORE retraining: statistics belong in the
        # embedding space the clients actually used to generate these tuples.
        for t in batch_tuples:
            apply_tuple_to_heads(self.backbone, self.item_head, self.price_head, self.nudge_head, t)

        train_loss_last_epoch = await asyncio.to_thread(
            retrain_backbone,
            self.backbone,
            self.reward_predictor,
            self._optimizer,
            self._tuple_pool,
        )

        loss_after = evaluate_backbone_loss(self.backbone, self.reward_predictor, self._tuple_pool)

        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_wall = time.perf_counter() - round_started_perf
        elapsed_cpu = time.process_time() - round_started_cpu
        training_duration_ms = int(round(elapsed_wall * 1000))
        cpu_usage_percentage = (elapsed_cpu / elapsed_wall) * 100 if elapsed_wall > 0 else 0.0
        memory_usage_mb = peak_memory_bytes / (1024 * 1024)
        loss_delta = float(loss_after - loss_before)

        model_version_before = self.model_version
        self.model_version += 1
        await self._persist_to_db(
            client_count=batch_clients,
            num_interactions=len(batch_tuples),
            contributing_client_ids=sorted(self._pending_uploads.keys()),
            training_duration_ms=training_duration_ms,
            model_version_before=model_version_before,
            cpu_usage_percentage=cpu_usage_percentage,
            memory_usage_mb=memory_usage_mb,
            loss_before=loss_before,
            loss_after=loss_after,
            loss_delta=loss_delta,
            timestamp=round_started_at,
        )

        self._rounds_completed += 1
        logger.info(
            "Centralized training round complete — version=%d clients=%d "
            "round_tuples=%d pool_size=%d loss_before=%.6f loss_after=%.6f "
            "loss_delta=%+.6f train_loss_last_epoch=%.6f training_duration_ms=%d "
            "rounds_completed=%d",
            self.model_version, batch_clients, len(batch_tuples), len(self._tuple_pool),
            loss_before, loss_after, loss_delta, train_loss_last_epoch,
            training_duration_ms, self._rounds_completed,
        )

        self._pending_uploads = {}

    def get_model_snapshot(self) -> dict[str, Any]:
        backbone_dict = _backbone_to_serialisable(self.backbone)
        reward_predictor_dict = {k: v.tolist() for k, v in self.reward_predictor.state_dict().items()}
        return {
            "version": self.model_version,
            "backbone_weights": _encode(backbone_dict),
            "reward_predictor_weights": _encode(reward_predictor_dict),
            "head_weights": {
                "item": _encode(self.item_head.state_dict()),
                "price": _encode(self.price_head.state_dict()),
                "nudge": _encode(self.nudge_head.state_dict()),
            },
        }
