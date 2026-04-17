"""
Centralized Training Service
=============================
Manages a centralized backbone + global heads for the centralized experiment arm.

- Receives raw interaction tuples from centralized-mode clients.
- Retrains the backbone on the accumulated tuple pool.
- Applies Bayesian online updates to global shared heads (item, price, nudge).
- Persists all state to the database so the server can restart without loss.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import random
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.db.models.centralized import CentralizedModelVersion
from app.logger import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RETRAIN_LR = 1e-3
RETRAIN_EPOCHS = 3
RETRAIN_BATCH_SIZE = 32
MAX_TUPLE_POOL_SIZE = 2000
CTX_PRICE_DELTA = 2  # index 2 in the 16-dim context vector

NUDGE_TYPES = ["N1", "N2", "N3", "N4"]


# ---------------------------------------------------------------------------
# Model architectures (exact copies of client-side models)
# ---------------------------------------------------------------------------
class BackboneEncoder(nn.Module):
    def __init__(self, input_dim: int = 16, latent_dim: int = 32):
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
def retrain_backbone(backbone: BackboneEncoder, reward_predictor: RewardPredictor,
                     tuples: list[dict]) -> float:
    if not tuples:
        return 0.0

    contexts = torch.tensor([t["context"] for t in tuples], dtype=torch.float32)
    rewards = torch.tensor([t["reward"] for t in tuples], dtype=torch.float32).unsqueeze(1)

    backbone.train()
    reward_predictor.train()
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(reward_predictor.parameters()),
        lr=RETRAIN_LR,
    )

    n = len(tuples)
    avg_loss = 0.0
    for _ in range(RETRAIN_EPOCHS):
        idx = list(range(n))
        random.shuffle(idx)
        for start in range(0, n, RETRAIN_BATCH_SIZE):
            batch = idx[start:start + RETRAIN_BATCH_SIZE]
            emb = backbone(contexts[batch])
            pred = reward_predictor(emb)
            loss = nn.functional.mse_loss(pred, rewards[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = float(loss.item())

    backbone.eval()
    reward_predictor.eval()
    return avg_loss


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

        # Tuple pool (bounded ring buffer via list + trim)
        self._tuple_pool: list[dict] = []

        # Versioning
        self.model_version: int = 0

    # ── Initialisation ─────────────────────────────────────────────────────

    def init_from_pretrained_weights(self, weights: dict[str, np.ndarray], version: int = 0):
        """
        Load backbone weights from the same pretrained checkpoint used by the
        federated arm.  `weights` is a dict of numpy arrays keyed by
        backbone parameter name (e.g. ``backbone.0.weight``).
        """
        sd = {
            k: torch.tensor(v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32),
                            dtype=torch.float32)
            for k, v in weights.items()
        }
        self.backbone.load_state_dict(sd)
        self.backbone.eval()
        self.model_version = version
        logger.info("Centralized backbone initialised from pretrained weights (version=%d).", version)

    async def try_load_persisted_state(self) -> bool:
        """
        Attempt to restore state from the database.
        Returns True if a persisted state was found and loaded.
        """
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(CentralizedModelVersion)
                    .order_by(CentralizedModelVersion.version.desc())
                    .limit(1)
                )
                row = result.scalar_one_or_none()

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
                "Centralized state restored from DB: version=%d, tuples=%d",
                self.model_version, len(self._tuple_pool),
            )
            return True

        except Exception:
            logger.exception("Failed to load persisted centralized state — will re-initialise.")
            return False

    async def _persist_to_db(self):
        """Insert a new CentralizedModelVersion row capturing the full current state."""
        backbone_blob = _encode(_backbone_to_serialisable(self.backbone))
        rp_blob = _encode({k: v.tolist() for k, v in self.reward_predictor.state_dict().items()})
        item_blob = _encode(self.item_head.state_dict())
        price_blob = _encode(self.price_head.state_dict())
        nudge_blob = _encode(self.nudge_head.state_dict())
        pool_blob = _encode(self._tuple_pool)

        async with AsyncSessionLocal() as db:
            row = CentralizedModelVersion(
                version=self.model_version,
                backbone_blob=backbone_blob,
                reward_predictor_blob=rp_blob,
                item_head_blob=item_blob,
                price_head_blob=price_blob,
                nudge_head_blob=nudge_blob,
                tuple_pool_blob=pool_blob,
            )
            db.add(row)
            await db.commit()

    # ── Core processing ────────────────────────────────────────────────────

    async def process_interactions(self, client_id: str, algorithm: str,
                                   count: int, data: str) -> int:
        """
        Decode a batch of interaction tuples, retrain the backbone, update
        global heads, bump model_version, and persist to the database.

        Returns the new model_version.
        """
        tuples = decode_tuples(data)
        logger.info(
            "Centralized: received %d tuples from client='%s' (algorithm=%s)",
            len(tuples), client_id, algorithm,
        )

        async with self._lock:
            # Append to pool, trim to window
            self._tuple_pool.extend(tuples)
            if len(self._tuple_pool) > MAX_TUPLE_POOL_SIZE:
                self._tuple_pool = self._tuple_pool[-MAX_TUPLE_POOL_SIZE:]

            # Retrain backbone on the pool
            loss = retrain_backbone(self.backbone, self.reward_predictor, self._tuple_pool)
            logger.info("Centralized backbone retrained: loss=%.6f, pool_size=%d",
                        loss, len(self._tuple_pool))

            # Update global heads with each tuple in order
            for t in tuples:
                apply_tuple_to_heads(self.backbone, self.item_head,
                                     self.price_head, self.nudge_head, t)

            # Bump version
            self.model_version += 1

            # Persist to DB
            await self._persist_to_db()
            logger.info("Centralized model updated to version=%d", self.model_version)

            return self.model_version

    # ── Model serialisation for GET endpoint ───────────────────────────────

    def get_model_snapshot(self) -> dict[str, Any]:
        """
        Build the response payload for GET /centralized/model.
        """
        backbone_dict = _backbone_to_serialisable(self.backbone)
        return {
            "model_version": self.model_version,
            "backbone_weights": _encode(backbone_dict),
            "head_weights": {
                "item": _encode(self.item_head.state_dict()),
                "price": _encode(self.price_head.state_dict()),
                "nudge": _encode(self.nudge_head.state_dict()),
            },
        }
