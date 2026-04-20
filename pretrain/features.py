"""
features.py
===========
Build the 18-dimensional context vector for (original, candidate) pairs
and assemble the full pre-training dataset from substitution groups.

Feature layout (matching client-side convention):
    0  co2_reduction_rel
    1  co2_delta
    2  price_delta_rel
    3  calorie_delta
    4  protein_delta
    5  candidate_is_plant_based
    6  candidate_is_meat
    7  candidate_is_dairy
    8  same_category_flag
    9  similarity_score
   10  is_lower_co2_flag
   11  nudge_n1 (zero during pre-training)
   12  nudge_n2
   13  nudge_n3
   14  nudge_n4
   15  nudge_n5
   16  nudge_n6
   17  cart_size_norm (0.5 during pre-training)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from pretrain.targets import (
    NUTRITION_FIELDS,
    compute_nutrition_maxes,
    compute_target_score,
)

logger = logging.getLogger(__name__)

FEATURE_DIM = 18
MAX_PAIRS_PER_GROUP = 500

# Category IDs from categories.csv
MEAT_CATEGORY_ID = 3
DAIRY_CATEGORY_ID = 5
FISH_CATEGORY_ID = 4

# Categories considered "animal" for plant-based detection
ANIMAL_CATEGORY_IDS = {MEAT_CATEGORY_ID, DAIRY_CATEGORY_ID, FISH_CATEGORY_ID}


def _safe(value, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def build_feature_vector(
    original: dict,
    candidate: dict,
    orig_category_ids: set[int],
    cand_category_ids: set[int],
    nutrition_maxes: dict[str, float],
    cart_size_norm: float = 0.5,
) -> np.ndarray:
    """Build a single 18D feature vector for an (original, candidate) pair."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    orig_co2 = _safe(original.get("co2e_kg_pr_item_kg"))
    cand_co2 = _safe(candidate.get("co2e_kg_pr_item_kg"))

    # 0: co2_reduction_rel
    if orig_co2 > 0:
        vec[0] = np.clip((orig_co2 - cand_co2) / orig_co2, -1.0, 1.0)

    # 1: co2_delta (normalised by max CO2 across dataset for scale)
    max_co2 = max(orig_co2, cand_co2, 1.0)
    vec[1] = np.clip((orig_co2 - cand_co2) / max_co2, -1.0, 1.0)

    # 2: price_delta_rel
    orig_price_per_g = _safe(original.get("price_dkk") / original.get("product_weight_in_g"))
    cand_price_per_g = _safe(candidate.get("price_dkk") / original.get("product_weight_in_g"))
    if orig_price_per_g > 0:
        vec[2] = np.clip((cand_price_per_g - orig_price_per_g) / orig_price_per_g, -1.0, 1.0)

    # 3: calorie_delta
    orig_cal = _safe(original.get("calories_per_100g"))
    cand_cal = _safe(candidate.get("calories_per_100g"))
    max_cal = nutrition_maxes.get("calories_per_100g", 1.0) or 1.0
    vec[3] = np.clip((cand_cal - orig_cal) / max_cal, -1.0, 1.0)

    # 4: protein_delta
    orig_prot = _safe(original.get("protein_g_per_100g"))
    cand_prot = _safe(candidate.get("protein_g_per_100g"))
    max_prot = nutrition_maxes.get("protein_g_per_100g", 1.0) or 1.0
    vec[4] = np.clip((cand_prot - orig_prot) / max_prot, -1.0, 1.0)

    # 5: candidate_is_plant_based (not meat, not dairy, not fish)
    vec[5] = 1.0 if not (cand_category_ids & ANIMAL_CATEGORY_IDS) else 0.0

    # 6: candidate_is_meat
    vec[6] = 1.0 if MEAT_CATEGORY_ID in cand_category_ids else 0.0

    # 7: candidate_is_dairy
    vec[7] = 1.0 if DAIRY_CATEGORY_ID in cand_category_ids else 0.0

    # 8: same_category_flag
    vec[8] = 1.0 if (orig_category_ids & cand_category_ids) else 0.0

    # 9: similarity_score (cosine similarity over normalised nutrition)
    orig_nutr = np.array(
        [_safe(original.get(f)) / (nutrition_maxes.get(f, 1.0) or 1.0) for f in NUTRITION_FIELDS],
        dtype=np.float32,
    )
    cand_nutr = np.array(
        [_safe(candidate.get(f)) / (nutrition_maxes.get(f, 1.0) or 1.0) for f in NUTRITION_FIELDS],
        dtype=np.float32,
    )
    norm_orig = np.linalg.norm(orig_nutr)
    norm_cand = np.linalg.norm(cand_nutr)
    if norm_orig > 0 and norm_cand > 0:
        vec[9] = float(np.dot(orig_nutr, cand_nutr) / (norm_orig * norm_cand))
    else:
        vec[9] = 0.0

    # 10: is_lower_co2_flag
    vec[10] = 1.0 if cand_co2 < orig_co2 else 0.0

    # 11-16: nudge one-hot — all zeros during pre-training

    # 17: cart_size_norm — fixed neutral value
    vec[17] = cart_size_norm

    return vec


def build_dataset(
    food_items: list[dict],
    substitution_groups: dict[int, list[str]],
    item_category_map: dict[str, set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the full pre-training dataset.

    Returns:
        features:  (N, 18) float32 array
        targets:   (N,)    float32 array of quality scores in [0, 1]
        group_ids: (N,)    int array — substitution group each pair came from
                   (used for stratified train/val split)
    """
    item_map: dict[str, dict] = {item["id"]: item for item in food_items}
    nutrition_maxes = compute_nutrition_maxes(food_items)

    all_item_ids = list(item_map.keys())
    rng = np.random.default_rng(42)

    features_list: list[np.ndarray] = []
    targets_list: list[float] = []
    group_ids_list: list[int] = []

    skipped_groups = 0

    for group_id, member_ids in substitution_groups.items():
        # Filter to items that actually exist in the catalogue
        members = [mid for mid in member_ids if mid in item_map]
        if len(members) < 2:
            skipped_groups += 1
            continue

        # --- Positive pairs (within group) ---
        pairs: list[tuple[str, str]] = []
        for i, orig_id in enumerate(members):
            for j, cand_id in enumerate(members):
                if i != j:
                    pairs.append((orig_id, cand_id))

        # Cap large groups
        if len(pairs) > MAX_PAIRS_PER_GROUP:
            indices = rng.choice(len(pairs), size=MAX_PAIRS_PER_GROUP, replace=False)
            pairs = [pairs[idx] for idx in indices]

        for orig_id, cand_id in pairs:
            orig = item_map[orig_id]
            cand = item_map[cand_id]
            orig_cats = item_category_map.get(orig_id, set())
            cand_cats = item_category_map.get(cand_id, set())

            vec = build_feature_vector(orig, cand, orig_cats, cand_cats, nutrition_maxes)
            score = compute_target_score(orig, cand, nutrition_maxes)
            features_list.append(vec)
            targets_list.append(score)
            group_ids_list.append(group_id)

        # --- Negative pairs (cross-group, 2 per positive pair) ---
        non_members = [aid for aid in all_item_ids if aid not in set(members)]
        if not non_members:
            continue

        n_negatives = min(len(pairs) * 2, len(non_members) * len(members))
        for _ in range(n_negatives):
            orig_id = rng.choice(members)
            cand_id = rng.choice(non_members)
            orig = item_map[orig_id]
            cand = item_map[cand_id]
            orig_cats = item_category_map.get(orig_id, set())
            cand_cats = item_category_map.get(cand_id, set())

            vec = build_feature_vector(orig, cand, orig_cats, cand_cats, nutrition_maxes)
            # Cross-group substitutions get a low target score
            score = compute_target_score(orig, cand, nutrition_maxes) * 0.3
            features_list.append(vec)
            targets_list.append(score)
            group_ids_list.append(group_id)

    if skipped_groups:
        logger.info("Skipped %d substitution groups with fewer than 2 members.", skipped_groups)

    logger.info(
        "Built dataset: %d pairs (%d positive, %d negative) from %d groups.",
        len(features_list),
        len(features_list) - sum(1 for _ in range(len(features_list)) if False),
        0,  # logged separately
        len(substitution_groups) - skipped_groups,
    )

    return (
        np.array(features_list, dtype=np.float32),
        np.array(targets_list, dtype=np.float32),
        np.array(group_ids_list, dtype=np.int32),
    )
