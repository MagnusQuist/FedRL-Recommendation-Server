"""
targets.py
==========
Compute synthetic recommendation quality scores for (original, candidate) pairs.

Score = 0.6 * co2_reward + 0.25 * nutrition_reward + 0.15 * price_reward
"""

from __future__ import annotations

import numpy as np

NUTRITION_FIELDS = (
    "calories_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "salt_g_per_100g",
)


def compute_nutrition_maxes(food_items: list[dict]) -> dict[str, float]:
    """Return per-field maximum values across all food items (for normalisation)."""
    maxes: dict[str, float] = {}
    for field in NUTRITION_FIELDS:
        values = [item[field] for item in food_items if item.get(field) is not None]
        maxes[field] = max(values) if values else 1.0
    return maxes


def _safe(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def compute_target_score(
    original: dict,
    candidate: dict,
    nutrition_maxes: dict[str, float],
) -> float:
    """
    Synthetic quality score in [0, 1].

    Higher when the candidate has lower CO2, similar nutrition, and reasonable price.
    """
    # --- CO2 reward (0.6 weight) ---
    orig_co2 = _safe(original.get("co2e_kg_pr_item_kg"))
    cand_co2 = _safe(candidate.get("co2e_kg_pr_item_kg"))

    if orig_co2 > 0 and cand_co2 > 0:
        raw = (orig_co2 - cand_co2) / orig_co2  # positive = candidate is greener
        co2_reward = (np.clip(raw, -1.0, 1.0) + 1.0) / 2.0
    else:
        co2_reward = 0.5  # neutral when data missing

    # --- Nutrition reward (0.25 weight) ---
    orig_vec = []
    cand_vec = []
    for field in NUTRITION_FIELDS:
        m = nutrition_maxes.get(field, 1.0) or 1.0
        orig_vec.append(_safe(original.get(field)) / m)
        cand_vec.append(_safe(candidate.get(field)) / m)

    orig_arr = np.array(orig_vec, dtype=np.float32)
    cand_arr = np.array(cand_vec, dtype=np.float32)
    dist = float(np.linalg.norm(orig_arr - cand_arr))
    max_dist = float(np.sqrt(len(NUTRITION_FIELDS)))  # theoretical max when normalised to [0,1]
    nutrition_reward = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)

    # --- Price reward (0.15 weight) ---
    orig_price = _safe(original.get("price_dkk"))
    cand_price = _safe(candidate.get("price_dkk"))

    if orig_price > 0:
        price_reward = float(np.clip(1.0 - abs(cand_price - orig_price) / orig_price, 0.0, 1.0))
    else:
        price_reward = 0.5  # neutral

    return float(0.6 * co2_reward + 0.25 * nutrition_reward + 0.15 * price_reward)
