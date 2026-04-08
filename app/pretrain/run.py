"""
run.py
======
Pre-train the backbone model and persist it to the database.

Usage:
    python -m app.pretrain.run [OPTIONS]

Options:
    --epochs INT       Maximum training epochs (default: 300)
    --lr FLOAT         Learning rate (default: 1e-3)
    --batch-size INT   Batch size (default: 256)
    --patience INT     Early stopping patience (default: 20)
    --seed INT         Random seed (default: 42)
    --plot             Show loss curves after training
    --save-plot PATH   Save loss curve plot to file
    --no-save          Dry run — train but don't write to DB
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import select, delete

from app.db import AsyncSessionLocal
from app.db.models.backbone import GlobalBackboneVersion
from app.db.models.food_item import FoodItem
from app.db.models.food_item_category import FoodItemCategory
from app.db.models.substitution_group_item import SubstitutionGroupItem
from app.db.seed_backbone import (
    SUPPORTED_ALGORITHMS,
    INITIAL_VERSION,
    serialise_weights,
)
from app.pretrain.features import build_dataset
from app.pretrain.model import BackboneWithHead
from app.pretrain.trainer import TrainConfig, train, plot_training

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train the recommendation backbone.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Show loss plot after training.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save loss plot to file.")
    parser.add_argument("--no-save", action="store_true", help="Dry run — skip DB write.")
    return parser.parse_args()


async def _load_data() -> tuple[list[dict], dict[int, list[str]], dict[str, set[int]]]:
    """Load food items, substitution groups, and category mappings from the DB."""
    async with AsyncSessionLocal() as db:
        # Food items
        result = await db.execute(select(FoodItem))
        food_items_raw = result.scalars().all()
        food_items = []
        for item in food_items_raw:
            food_items.append({
                "id": item.id,
                "name": item.name,
                "brand": item.brand,
                "product_weight_in_g": item.product_weight_in_g,
                "co2e_kg_pr_item_kg": float(item.co2e_kg_pr_item_kg) if item.co2e_kg_pr_item_kg is not None else None,
                "estimated_co2e_kg_pr_item_weight_in_g": float(item.estimated_co2e_kg_pr_item_weight_in_g) if item.estimated_co2e_kg_pr_item_weight_in_g is not None else None,
                "calories_per_100g": item.calories_per_100g,
                "protein_g_per_100g": float(item.protein_g_per_100g) if item.protein_g_per_100g is not None else None,
                "fat_g_per_100g": float(item.fat_g_per_100g) if item.fat_g_per_100g is not None else None,
                "carbs_g_per_100g": float(item.carbs_g_per_100g) if item.carbs_g_per_100g is not None else None,
                "fiber_g_per_100g": float(item.fiber_g_per_100g) if item.fiber_g_per_100g is not None else None,
                "salt_g_per_100g": float(item.salt_g_per_100g) if item.salt_g_per_100g is not None else None,
                "price_dkk": float(item.price_dkk) if item.price_dkk is not None else None,
            })

        # Substitution groups: group_id -> [product_id, ...]
        result = await db.execute(select(SubstitutionGroupItem))
        sub_rows = result.scalars().all()
        substitution_groups: dict[int, list[str]] = defaultdict(list)
        for row in sub_rows:
            substitution_groups[row.substitution_group_id].append(row.product_id)

        # Category map: product_id -> {category_id, ...}
        result = await db.execute(select(FoodItemCategory))
        cat_rows = result.scalars().all()
        item_category_map: dict[str, set[int]] = defaultdict(set)
        for row in cat_rows:
            item_category_map[row.product_id].add(row.category_id)

    logger.info(
        "Loaded %d food items, %d substitution groups, %d category mappings.",
        len(food_items),
        len(substitution_groups),
        sum(len(v) for v in item_category_map.values()),
    )
    return food_items, dict(substitution_groups), dict(item_category_map)


async def _save_backbone(weights: dict[str, np.ndarray], dry_run: bool) -> None:
    """Persist pre-trained weights as version 1 for each algorithm."""
    if dry_run:
        logger.info("--no-save: skipping DB write.")
        return

    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        for algorithm in SUPPORTED_ALGORITHMS:
            # Remove existing version 1 if present
            await db.execute(
                delete(GlobalBackboneVersion).where(
                    GlobalBackboneVersion.algorithm == algorithm,
                    GlobalBackboneVersion.version == INITIAL_VERSION,
                )
            )

            backbone = GlobalBackboneVersion(
                version=INITIAL_VERSION,
                weights_blob=blob,
                algorithm=algorithm,
                client_count=0,
                total_interactions=0,
            )
            db.add(backbone)
            logger.info("Saved pre-trained backbone version=%d for algorithm='%s'.", INITIAL_VERSION, algorithm)

        await db.commit()

    logger.info("Pre-trained backbone persisted for algorithms: %s", list(SUPPORTED_ALGORITHMS))


def _stratified_split(
    group_ids: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so that all pairs from a group go to the same fold."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(group_ids)
    rng.shuffle(unique_groups)

    split_idx = int(len(unique_groups) * train_ratio)
    train_groups = set(unique_groups[:split_idx].tolist())

    all_indices = np.arange(len(group_ids))
    train_mask = np.array([gid in train_groups for gid in group_ids])

    return all_indices[train_mask], all_indices[~train_mask]


async def _run(args: argparse.Namespace) -> None:
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    food_items, substitution_groups, item_category_map = await _load_data()
    if not food_items:
        logger.error("No food items in database. Run seed_catalogue first.")
        sys.exit(1)

    # Build dataset
    features, targets, group_ids = build_dataset(food_items, substitution_groups, item_category_map)
    logger.info("Dataset: %d samples, feature_dim=%d", len(features), features.shape[1])

    # Stratified split
    train_idx, val_idx = _stratified_split(group_ids, seed=args.seed)
    logger.info("Train: %d samples, Val: %d samples", len(train_idx), len(val_idx))

    train_ds = TensorDataset(
        torch.from_numpy(features[train_idx]),
        torch.from_numpy(targets[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(features[val_idx]),
        torch.from_numpy(targets[val_idx]),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Train
    model = BackboneWithHead()
    config = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )
    logger.info("Starting pre-training: epochs=%d, lr=%s, batch_size=%d, patience=%d",
                config.epochs, config.lr, args.batch_size, config.patience)

    history = train(model, train_loader, val_loader, config)

    logger.info(
        "Training complete. Best epoch=%d, best_val_loss=%.6f",
        history.best_epoch,
        min(history.val_loss) if history.val_loss else float("nan"),
    )

    # Plot
    if args.plot or args.save_plot:
        plot_training(history, save_path=args.save_plot)

    # Save
    weights = model.get_backbone_weights_numpy()
    for key, arr in weights.items():
        logger.info("  %s  shape=%s  dtype=%s", key, arr.shape, arr.dtype)

    await _save_backbone(weights, dry_run=args.no_save)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
