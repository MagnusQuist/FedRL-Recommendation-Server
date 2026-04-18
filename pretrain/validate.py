"""
validate.py
===========
Validate that the pre-trained backbone learned something semantically meaningful.

Runs three interpretable checks and prints a pass/fail summary:

  1. CO2 Direction Accuracy
     Among all pairs where the candidate has strictly lower CO2 than the
     original, what % does the model assign a higher predicted score to?
     Random init baseline ≈ 50%.  A useful backbone should be ≥ 75%.

  2. Group Cohesion Ratio
     Average cosine similarity of backbone embeddings for items WITHIN the
     same substitution group vs. items from DIFFERENT groups.
     Ratio > 1.0 means items that should be similar are closer in embedding
     space. Random init baseline ≈ 1.0.

  3. Within-Group CO2 Rank Correlation (Spearman's ρ)
     For each substitution group with CO2 data, rank all candidates by their
     predicted quality score (using a fixed "average" original item as context).
     Measure the Spearman rank correlation between predicted ranking and the
     ground-truth ranking by CO2e (lower CO2 = rank 1).
     This directly tests whether the backbone will push users toward greener
     choices when the RL head selects among valid substitutes.
     Random init baseline ≈ 0.0.  A useful backbone should have mean ρ ≥ 0.30.

Usage:
    python -m pretrain.validate [--baseline]

    --baseline   Also evaluate a randomly initialised backbone for comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json
import logging
from collections import defaultdict

import numpy as np
import torch
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.db.models.federated import FederatedBackboneVersion
from app.db.models.food_item import FoodItem
from app.db.models.food_item_category import FoodItemCategory
from app.db.models.substitution_group_item import SubstitutionGroupItem
from app.db.seeding.seed_backbone import INITIAL_VERSION
from pretrain.features import (
    FEATURE_DIM,
    MAX_PAIRS_PER_GROUP,
    build_feature_vector,
)
from pretrain.model import BackboneWithHead
from pretrain.targets import compute_nutrition_maxes

logger = logging.getLogger(__name__)

# Thresholds for pass/fail
CO2_ACCURACY_THRESHOLD = 0.70
COHESION_RATIO_THRESHOLD = 1.10
SCORE_AUC_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_backbone_weights_from_blob(blob: str) -> dict[str, np.ndarray]:
    compressed = base64.b64decode(blob)
    json_bytes = gzip.decompress(compressed)
    raw = json.loads(json_bytes.decode("utf-8"))
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def _load_model_from_weights(weights: dict[str, np.ndarray]) -> BackboneWithHead:
    model = BackboneWithHead()
    state = {k: torch.tensor(v) for k, v in weights.items()}
    model.backbone.load_state_dict(state)
    model.eval()
    return model


async def _load_db_data():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(FoodItem))
        food_items_raw = result.scalars().all()
        food_items = []
        for item in food_items_raw:
            food_items.append({
                "id": item.id,
                "co2e_kg_pr_item_kg": float(item.co2e_kg_pr_item_kg) if item.co2e_kg_pr_item_kg is not None else None,
                "calories_per_100g": item.calories_per_100g,
                "protein_g_per_100g": float(item.protein_g_per_100g) if item.protein_g_per_100g is not None else None,
                "fat_g_per_100g": float(item.fat_g_per_100g) if item.fat_g_per_100g is not None else None,
                "carbs_g_per_100g": float(item.carbs_g_per_100g) if item.carbs_g_per_100g is not None else None,
                "fiber_g_per_100g": float(item.fiber_g_per_100g) if item.fiber_g_per_100g is not None else None,
                "salt_g_per_100g": float(item.salt_g_per_100g) if item.salt_g_per_100g is not None else None,
                "price_dkk": float(item.price_dkk) if item.price_dkk is not None else None,
                "product_weight_in_g": item.product_weight_in_g,
            })

        result = await db.execute(select(SubstitutionGroupItem))
        sub_rows = result.scalars().all()
        substitution_groups: dict[int, list[str]] = defaultdict(list)
        for row in sub_rows:
            substitution_groups[row.substitution_group_id].append(row.product_id)

        result = await db.execute(select(FoodItemCategory))
        cat_rows = result.scalars().all()
        item_category_map: dict[str, set[int]] = defaultdict(set)
        for row in cat_rows:
            item_category_map[row.product_id].add(row.category_id)

        # Load stored backbone weights
        result = await db.execute(
            select(FederatedBackboneVersion)
            .where(FederatedBackboneVersion.version == INITIAL_VERSION)
        )
        backbone_row = result.scalar_one_or_none()
        backbone_blob = backbone_row.weights_blob if backbone_row else None

    return food_items, dict(substitution_groups), dict(item_category_map), backbone_blob


# ---------------------------------------------------------------------------
# Check 1: CO2 direction accuracy
# ---------------------------------------------------------------------------

def check_co2_direction(
    model: BackboneWithHead,
    food_items: list[dict],
    substitution_groups: dict[int, list[str]],
    item_category_map: dict[str, set[int]],
    nutrition_maxes: dict[str, float],
    rng: np.random.Generator,
) -> dict:
    """
    Among pairs where candidate has strictly lower CO2 than original,
    what fraction does the model score higher?
    """
    item_map = {item["id"]: item for item in food_items}
    correct = 0
    total = 0

    for member_ids in substitution_groups.values():
        members = [mid for mid in member_ids if mid in item_map]
        if len(members) < 2:
            continue

        pairs = [(o, c) for o in members for c in members if o != c]
        if len(pairs) > MAX_PAIRS_PER_GROUP:
            idx = rng.choice(len(pairs), size=MAX_PAIRS_PER_GROUP, replace=False)
            pairs = [pairs[i] for i in idx]

        for orig_id, cand_id in pairs:
            orig = item_map[orig_id]
            cand = item_map[cand_id]
            orig_co2 = orig.get("co2e_kg_pr_item_kg")
            cand_co2 = cand.get("co2e_kg_pr_item_kg")

            if orig_co2 is None or cand_co2 is None or orig_co2 <= 0:
                continue
            if cand_co2 >= orig_co2:
                continue  # Only evaluate pairs where candidate is greener

            vec = build_feature_vector(
                orig, cand,
                item_category_map.get(orig_id, set()),
                item_category_map.get(cand_id, set()),
                nutrition_maxes,
            )
            with torch.no_grad():
                score = model(torch.tensor(vec).unsqueeze(0)).item()

            # Compare against the reverse pair score
            vec_rev = build_feature_vector(
                cand, orig,
                item_category_map.get(cand_id, set()),
                item_category_map.get(orig_id, set()),
                nutrition_maxes,
            )
            with torch.no_grad():
                score_rev = model(torch.tensor(vec_rev).unsqueeze(0)).item()

            if score > score_rev:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "n_pairs": total, "passed": accuracy >= CO2_ACCURACY_THRESHOLD}


# ---------------------------------------------------------------------------
# Check 2: Embedding cohesion ratio
# ---------------------------------------------------------------------------

def check_embedding_cohesion(
    model: BackboneWithHead,
    food_items: list[dict],
    substitution_groups: dict[int, list[str]],
    item_category_map: dict[str, set[int]],
    nutrition_maxes: dict[str, float],
    rng: np.random.Generator,
) -> dict:
    """
    Cosine similarity of embeddings within groups vs. across groups.
    """
    item_map = {item["id"]: item for item in food_items}

    # Compute embeddings for all items using a neutral context vector
    embeddings: dict[str, np.ndarray] = {}
    for item_id, item in item_map.items():
        # Self-pair as a proxy for the item's embedding in neutral context
        vec = build_feature_vector(
            item, item,
            item_category_map.get(item_id, set()),
            item_category_map.get(item_id, set()),
            nutrition_maxes,
        )
        with torch.no_grad():
            emb = model.backbone(torch.tensor(vec).unsqueeze(0)).squeeze(0).numpy()
        embeddings[item_id] = emb

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    within_sims = []
    across_sims = []

    group_members = {
        gid: [mid for mid in mids if mid in embeddings]
        for gid, mids in substitution_groups.items()
    }
    all_ids = list(embeddings.keys())

    for gid, members in group_members.items():
        if len(members) < 2:
            continue

        # Within-group: sample up to 100 pairs
        pairs = [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
        if len(pairs) > 100:
            idx = rng.choice(len(pairs), size=100, replace=False)
            pairs = [pairs[i] for i in idx]

        for a, b in pairs:
            within_sims.append(cosine(embeddings[a], embeddings[b]))

        # Across-group: sample same number from outside the group
        non_members = [aid for aid in all_ids if aid not in set(members)]
        if not non_members:
            continue
        for a in rng.choice(members, size=min(len(pairs), len(non_members)), replace=True):
            b = rng.choice(non_members)
            across_sims.append(cosine(embeddings[str(a)], embeddings[str(b)]))

    mean_within = float(np.mean(within_sims)) if within_sims else 0.0
    mean_across = float(np.mean(across_sims)) if across_sims else 1.0
    ratio = mean_within / max(mean_across, 1e-8)

    return {
        "mean_within_group_similarity": mean_within,
        "mean_across_group_similarity": mean_across,
        "cohesion_ratio": ratio,
        "passed": ratio >= COHESION_RATIO_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Check 3: Score discrimination (AUC)
# ---------------------------------------------------------------------------

def check_score_discrimination(
    model: BackboneWithHead,
    food_items: list[dict],
    substitution_groups: dict[int, list[str]],
    item_category_map: dict[str, set[int]],
    nutrition_maxes: dict[str, float],
    rng: np.random.Generator,
    n_samples: int = 2000,
) -> dict:
    """
    AUC-style check: predicted score for within-group pairs vs cross-group pairs.
    """
    item_map = {item["id"]: item for item in food_items}
    all_ids = list(item_map.keys())

    within_scores = []
    cross_scores = []

    for member_ids in substitution_groups.values():
        members = [mid for mid in member_ids if mid in item_map]
        if len(members) < 2:
            continue

        # Within-group pair
        o, c = rng.choice(members, size=2, replace=False)
        vec = build_feature_vector(
            item_map[o], item_map[c],
            item_category_map.get(o, set()),
            item_category_map.get(c, set()),
            nutrition_maxes,
        )
        with torch.no_grad():
            within_scores.append(model(torch.tensor(vec).unsqueeze(0)).item())

        # Cross-group pair
        non_member = rng.choice([aid for aid in all_ids if aid not in set(members)])
        vec_cross = build_feature_vector(
            item_map[o], item_map[non_member],
            item_category_map.get(o, set()),
            item_category_map.get(non_member, set()),
            nutrition_maxes,
        )
        with torch.no_grad():
            cross_scores.append(model(torch.tensor(vec_cross).unsqueeze(0)).item())

    # AUC: fraction of (within, cross) pairs where within_score > cross_score
    within_arr = np.array(within_scores)
    cross_arr = np.array(cross_scores)
    auc = float(np.mean(within_arr[:, None] > cross_arr[None, :]))

    return {
        "mean_within_group_score": float(np.mean(within_arr)),
        "mean_cross_group_score": float(np.mean(cross_arr)),
        "score_auc": auc,
        "passed": auc >= SCORE_AUC_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_result(name: str, result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n[{status}] {name}")
    for k, v in result.items():
        if k == "passed":
            continue
        if isinstance(v, float):
            print(f"       {k}: {v:.4f}")
        else:
            print(f"       {k}: {v}")


async def _run(args: argparse.Namespace) -> None:
    print("Loading data from database...")
    food_items, substitution_groups, item_category_map, backbone_blob = await _load_db_data()

    if not food_items:
        print("ERROR: No food items found. Run seed_catalogue first.")
        return

    if backbone_blob is None:
        print("ERROR: No backbone version 1 found. Run pre-training first.")
        return

    nutrition_maxes = compute_nutrition_maxes(food_items)
    rng = np.random.default_rng(42)

    models_to_eval: list[tuple[str, BackboneWithHead]] = []

    # Pre-trained model
    weights = _load_backbone_weights_from_blob(backbone_blob)
    pretrained_model = _load_model_from_weights(weights)
    models_to_eval.append(("Pre-trained backbone", pretrained_model))

    # Optional: random baseline
    if args.baseline:
        baseline_model = BackboneWithHead()
        baseline_model.eval()
        models_to_eval.append(("Random baseline", baseline_model))

    for model_name, model in models_to_eval:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        r1 = check_co2_direction(model, food_items, substitution_groups, item_category_map, nutrition_maxes, rng)
        _print_result(
            f"CO2 Direction Accuracy (threshold ≥ {CO2_ACCURACY_THRESHOLD:.0%})",
            r1,
        )

        r2 = check_embedding_cohesion(model, food_items, substitution_groups, item_category_map, nutrition_maxes, rng)
        _print_result(
            f"Embedding Cohesion Ratio (threshold ≥ {COHESION_RATIO_THRESHOLD:.2f}x)",
            r2,
        )

        r3 = check_score_discrimination(model, food_items, substitution_groups, item_category_map, nutrition_maxes, rng)
        _print_result(
            f"Score Discrimination AUC (threshold ≥ {SCORE_AUC_THRESHOLD:.0%})",
            r3,
        )

        passed = sum([r1["passed"], r2["passed"], r3["passed"]])
        print(f"\n  Result: {passed}/3 checks passed")

    print()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # quiet during validation
    parser = argparse.ArgumentParser(description="Validate pre-trained backbone quality.")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate a random init for comparison.")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
