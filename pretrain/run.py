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
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import select, delete

from app.db import AsyncSessionLocal
from app.db.models.federated import FederatedModel
from app.db.models.food_item import FoodItem
from app.db.models.food_item_category import FoodItemCategory
from app.db.models.substitution_group_item import SubstitutionGroupItem
from app.db.seeding.seed_backbone import (
    INITIAL_VERSION,
    serialise_weights,
)
from pretrain.features import build_dataset
from pretrain.model import BackboneWithHead
from pretrain.trainer import TrainConfig, train, plot_training

logger = logging.getLogger(__name__)


@dataclass
class PretrainOptions:
    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 256
    patience: int = 20
    seed: int = 42
    plot: bool = False
    save_plot: str | None = None
    no_save: bool = False
    output_dir: str | None = None


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for training artifacts (weights/plot/summary).",
    )
    return parser.parse_args()


def options_from_args(args: argparse.Namespace) -> PretrainOptions:
    return PretrainOptions(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        plot=args.plot,
        save_plot=args.save_plot,
        no_save=args.no_save,
        output_dir=args.output_dir,
    )


def _prepare_artifact_paths(options: PretrainOptions) -> tuple[Path | None, Path | None]:
    """Create output directory and default loss-plot path when configured."""
    if not options.output_dir:
        return None, Path(options.save_plot) if options.save_plot else None

    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plot_path = Path(options.save_plot) if options.save_plot else output_dir / "training_loss.png"
    return output_dir, save_plot_path


def _save_artifacts(
    output_dir: Path | None,
    weights: dict[str, np.ndarray],
    result: dict[str, Any],
    options: PretrainOptions,
) -> dict[str, str | None]:
    """Persist local training artifacts for run-to-run comparison."""
    if output_dir is None:
        return {"artifacts_dir": None, "weights_file": None, "summary_file": None}

    weights_path = output_dir / "backbone_weights.npz"
    np.savez_compressed(weights_path, **weights)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "options": {
            "epochs": options.epochs,
            "lr": options.lr,
            "batch_size": options.batch_size,
            "patience": options.patience,
            "seed": options.seed,
            "no_save": options.no_save,
        },
        "result": result,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.info("Saved training artifacts to %s", output_dir)
    return {
        "artifacts_dir": str(output_dir),
        "weights_file": str(weights_path),
        "summary_file": str(summary_path),
    }


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
    """Persist pre-trained weights as the initial federated backbone (version 1)."""
    if dry_run:
        logger.info("--no-save: skipping DB write.")
        return

    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        await db.execute(
            delete(FederatedModel).where(
                FederatedModel.version == INITIAL_VERSION,
            )
        )

        backbone = FederatedModel(
            version=INITIAL_VERSION,
            weights_blob=blob,
        )
        db.add(backbone)
        await db.commit()

    logger.info(
        "Pre-trained backbone persisted: version=%d.",
        INITIAL_VERSION,
    )


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


async def run_pretraining(options: PretrainOptions) -> dict[str, Any]:
    output_dir, save_plot_path = _prepare_artifact_paths(options)

    # Reproducibility
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)

    # Load data
    food_items, substitution_groups, item_category_map = await _load_data()
    if not food_items:
        logger.error("No food items in database. Run seed_catalogue first.")
        sys.exit(1)

    # Build dataset
    features, targets, group_ids = build_dataset(food_items, substitution_groups, item_category_map)
    logger.info("Dataset: %d samples, feature_dim=%d", len(features), features.shape[1])

    # Stratified split
    train_idx, val_idx = _stratified_split(group_ids, seed=options.seed)
    logger.info("Train: %d samples, Val: %d samples", len(train_idx), len(val_idx))

    train_ds = TensorDataset(
        torch.from_numpy(features[train_idx]),
        torch.from_numpy(targets[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(features[val_idx]),
        torch.from_numpy(targets[val_idx]),
    )

    train_loader = DataLoader(train_ds, batch_size=options.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=options.batch_size)

    # Train
    model = BackboneWithHead()
    config = TrainConfig(
        lr=options.lr,
        epochs=options.epochs,
        patience=options.patience,
    )
    logger.info("Starting pre-training: epochs=%d, lr=%s, batch_size=%d, patience=%d",
                config.epochs, config.lr, options.batch_size, config.patience)

    history = train(model, train_loader, val_loader, config)

    logger.info(
        "Training complete. Best epoch=%d, best_val_loss=%.6f",
        history.best_epoch,
        min(history.val_loss) if history.val_loss else float("nan"),
    )

    # Plot
    if options.plot or save_plot_path:
        plot_training(history, save_path=str(save_plot_path) if save_plot_path else None)

    # Save
    weights = model.get_backbone_weights_numpy()
    for key, arr in weights.items():
        logger.info("  %s  shape=%s  dtype=%s", key, arr.shape, arr.dtype)

    await _save_backbone(weights, dry_run=options.no_save)

    result = {
        "dataset_samples": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_epoch": int(history.best_epoch),
        "best_val_loss": float(min(history.val_loss) if history.val_loss else float("nan")),
        "saved_to_db": not options.no_save,
        "save_plot": str(save_plot_path) if save_plot_path else None,
    }
    result.update(_save_artifacts(output_dir, weights, result, options))
    return result


async def _run(args: argparse.Namespace) -> None:
    options = options_from_args(args)
    await run_pretraining(options)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
