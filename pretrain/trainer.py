"""
trainer.py
==========
Training loop with early stopping, LR scheduling, and optional loss plotting.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-5
    device: str = "cpu"


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = 0


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.best_state: dict | None = None
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> TrainHistory:
    """Run the full training loop. Returns loss history."""
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    history = TrainHistory()

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        model.train()
        train_total = 0.0
        train_count = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(features)
            train_count += len(features)

        train_loss = train_total / max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                preds = model(features)
                loss = criterion(preds, targets)
                val_total += loss.item() * len(features)
                val_count += len(features)

        val_loss = val_total / max(val_count, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(current_lr)

        scheduler.step(val_loss)

        if early_stop.best_loss is None or val_loss < early_stop.best_loss:
            history.best_epoch = epoch

        should_stop = early_stop.step(val_loss, model)

        if epoch % 10 == 0 or epoch == 1 or should_stop:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e%s",
                epoch,
                config.epochs,
                train_loss,
                val_loss,
                current_lr,
                "  *best*" if epoch == history.best_epoch else "",
            )

        if should_stop:
            logger.info(
                "Early stopping at epoch %d. Best val_loss=%.6f at epoch %d.",
                epoch,
                early_stop.best_loss,
                history.best_epoch,
            )
            break

    # Restore best weights
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
        logger.info("Restored best model weights from epoch %d.", history.best_epoch)

    return history


def plot_training(history: TrainHistory, save_path: str | None = None) -> None:
    """Plot train/val loss curves. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.train_loss) + 1)
    ax1.plot(epochs, history.train_loss, label="Train")
    ax1.plot(epochs, history.val_loss, label="Validation")
    ax1.axvline(history.best_epoch, color="gray", linestyle="--", alpha=0.5, label="Best epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.lr)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Saved training plot to %s", save_path)
    else:
        plt.show()
    # Prevent matplotlib figure accumulation during sweep runs.
    plt.close(fig)
