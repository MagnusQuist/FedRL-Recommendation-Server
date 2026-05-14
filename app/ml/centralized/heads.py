"""Centralized Thompson-sampling heads."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from app.ml.centralized.models import BackboneEncoder

CTX_PRICE_DELTA = 2
NUDGE_TYPES = ["N1", "N2", "N3", "N4", "N5", "N6"]


class TSItemHead:
    def __init__(self, latent_dim=32, lam=1.0, v=0.5, max_items=200):
        self.latent_dim = latent_dim
        self.lam = lam
        self.v = v
        self.max_items = max_items
        self._params: OrderedDict[str, dict] = OrderedDict()

    def _init_item(self, item_id):
        if len(self._params) >= self.max_items:
            self._params.popitem(last=False)
        entry = {"A": np.eye(self.latent_dim) * self.lam, "b": np.zeros(self.latent_dim)}
        self._params[item_id] = entry
        return entry

    def _touch(self, item_id):
        if item_id not in self._params:
            return self._init_item(item_id)
        self._params.move_to_end(item_id)
        return self._params[item_id]

    def update(self, item_id: str, embedding: np.ndarray, reward: float):
        params = self._touch(item_id)
        params["A"] += np.outer(embedding, embedding)
        params["b"] += reward * embedding

    def state_dict(self) -> dict:
        return {
            "params": {
                key: {"A": value["A"].tolist(), "b": value["b"].tolist()}
                for key, value in self._params.items()
            },
            "lam": self.lam,
            "v": self.v,
            "max_items": self.max_items,
        }

    def load_state_dict(self, state: dict):
        self.lam = state.get("lam", self.lam)
        self.v = state.get("v", self.v)
        self.max_items = state.get("max_items", self.max_items)
        self._params = OrderedDict()
        for key, value in state.get("params", {}).items():
            self._params[key] = {
                "A": np.array(value["A"]),
                "b": np.array(value["b"]),
            }


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

    def load_state_dict(self, state: dict):
        self._A = state["A"]
        self._b = state["b"]
        self.lam = state.get("lam", self.lam)
        self.v = state.get("v", self.v)


class TSNudgeHead:
    def __init__(self, prior_mu=0.5, prior_tau=1.0):
        self._params = {
            nudge: {"mu": prior_mu, "tau": prior_tau, "sum": 0.0, "count": 0}
            for nudge in NUDGE_TYPES
        }
        self._interaction_count = 0

    def update(self, nudge_type: str, reward: float):
        params = self._params[nudge_type]
        tau_obs = 1.0
        params["count"] += 1
        params["sum"] += reward
        params["tau"] += tau_obs
        params["mu"] = params["sum"] * tau_obs / params["tau"]
        self._interaction_count += 1

    def state_dict(self):
        return {
            "params": self._params,
            "interaction_count": self._interaction_count,
            "rr_index": 0,
            "last_reward": 0.0,
        }

    def load_state_dict(self, state: dict):
        self._params = state["params"]
        self._interaction_count = state["interaction_count"]


def apply_tuple_to_heads(
    backbone: BackboneEncoder,
    item_head: TSItemHead,
    price_head: TSPriceHead,
    nudge_head: TSNudgeHead,
    interaction: dict,
):
    context = interaction["context"]
    embedding = backbone.embed(context).numpy()
    price_delta = context[CTX_PRICE_DELTA] if len(context) > CTX_PRICE_DELTA else 0.0
    item_head.update(interaction["alternative_id"], embedding, interaction["reward"])
    price_head.update(price_delta, interaction["reward"])
    nudge_head.update(interaction["nudge_type"], interaction["reward"])
