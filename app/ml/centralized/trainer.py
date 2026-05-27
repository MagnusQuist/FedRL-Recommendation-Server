"""Pure centralized training and evaluation functions."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.logging import logger
from app.ml.centralized.models import BackboneEncoder, RewardPredictor

RETRAIN_EPOCHS = 5
RETRAIN_BATCH_SIZE = 64
RETRAIN_GRAD_CLIP = 1.0


def flat_l2_norm_state_dict(state: dict[str, torch.Tensor]) -> float:
    """L2 norm of all parameter tensors in a state_dict flattened into one vector."""
    total = sum(t.detach().cpu().double().pow(2).sum().item() for t in state.values())
    return float(total ** 0.5)


def diff_l2_norm_state_dict(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    """L2 norm of (after[k] - before[k]) across all parameter tensors."""
    total = sum(
        (after[k].detach().cpu().double() - before[k].detach().cpu().double()).pow(2).sum().item()
        for k in before
    )
    return float(total ** 0.5)


def retrain_backbone(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
    optimizer: optim.Optimizer,
    tuples: list[dict],
    seed: int | None = None,
) -> float:
    """
    Train backbone + reward_predictor jointly with BCE on reward > 0 labels.
    The optimizer is supplied by the service so Adam state persists across rounds.
    """
    if not tuples:
        return 0.0

    device = next(backbone.parameters()).device
    rng = random.Random(seed)
    contexts = torch.tensor(
        [interaction["context"] for interaction in tuples],
        dtype=torch.float32,
        device=device,
    )
    labels = (
        torch.tensor(
            [interaction["reward"] for interaction in tuples],
            dtype=torch.float32,
            device=device,
        )
        > 0
    ).float().unsqueeze(1)

    n_samples = len(tuples)
    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        initial_embeddings = backbone(contexts)
        initial_logits = reward_predictor(initial_embeddings)
        initial_loss = nn.functional.binary_cross_entropy_with_logits(
            initial_logits,
            labels,
        ).item()
        initial_acc = float(((initial_logits > 0) == labels.bool()).float().mean().item())
        initial_emb_norm = float(initial_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain start: bce=%.4f acc=%.3f emb_norm=%.3f (%d tuples)",
        initial_loss,
        initial_acc,
        initial_emb_norm,
        n_samples,
    )

    backbone.train()
    reward_predictor.train()
    params = [param for group in optimizer.param_groups for param in group["params"]]

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
            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=RETRAIN_GRAD_CLIP,
            )
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
            epoch + 1,
            RETRAIN_EPOCHS,
            final_epoch_loss,
            final_epoch_acc,
            final_grad_norm_pre_clip,
            n_batches,
        )

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        final_embeddings = backbone(contexts)
        final_emb_norm = float(final_embeddings.norm(dim=1).mean().item())

    logger.info(
        "centralized_retrain done: bce %.4f -> %.4f, acc %.3f -> %.3f, "
        "emb_norm %.3f -> %.3f, avg_grad_norm=%.3f",
        initial_loss,
        final_epoch_loss,
        initial_acc,
        final_epoch_acc,
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
    if not tuples:
        return 0.0

    contexts = torch.tensor(
        [interaction["context"] for interaction in tuples],
        dtype=torch.float32,
    )
    labels = (
        torch.tensor(
            [interaction["reward"] for interaction in tuples],
            dtype=torch.float32,
        )
        > 0
    ).float().unsqueeze(1)

    backbone.eval()
    reward_predictor.eval()
    with torch.no_grad():
        embeddings = backbone(contexts)
        logits = reward_predictor(embeddings)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return float(loss.item())


def reward_stats(tuples: list[dict]) -> dict[str, float]:
    if not tuples:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    rewards = np.asarray([interaction["reward"] for interaction in tuples], dtype=np.float32)
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "n": int(rewards.size),
    }
