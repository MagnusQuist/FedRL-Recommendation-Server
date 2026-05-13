"""
Standalone trainer for the centralized backbone + reward predictor.

Decoupled from the rest of the application by design — the model
architectures are copied inline rather than imported from
``app/backbones/centralized.py``. That keeps the script runnable as a
single file (no DB, no async, no FastAPI) and pinned to a known
architecture so hyperparameter experiments stay reproducible.

Subcommands
-----------
generate
    Synthesize an interactions JSON file. Each simulated client has
    its own linear preference vector over the 21-dim context, so the
    backbone has a real signal to learn. Reward shape mirrors the
    observed real-data regime: a dismiss floor at -0.3 or an accept
    value in roughly [0.3, 0.95].

train
    Run one training pass on a CSV (real-data format) or JSON
    (generated) interactions file, or on a directory containing one or
    more such files (one per client, all concatenated into the training
    pool). Holds out a validation slice for an honest generalization
    signal. Writes the full hyperparameter set, per-epoch metric
    history, final metrics, and reward stats to a results JSON.

compare
    Read N results JSONs and print a side-by-side table so different
    runs can be diffed at a glance.

Examples
--------
    python scripts/train_centralized_standalone.py generate \
        --n-clients 50 --interactions-per-client 100 \
        --out data/synth.json

    python scripts/train_centralized_standalone.py train \
        --data extracted_interactions.csv --run-name baseline

    python scripts/train_centralized_standalone.py train \
        --data clients/ --run-name baseline_all_clients

    python scripts/train_centralized_standalone.py train \
        --data clients/ --run-name hi_pred_lr \
        --lr-backbone 1e-4 --lr-predictor 1e-3 --epochs 20 \
        --weight-decay 0.0

    python scripts/train_centralized_standalone.py train \
        --data clients/ --run-name bce_pooled \
        --objective bce --epochs 20

    python scripts/train_centralized_standalone.py compare \
        results/baseline.json results/hi_pred_lr.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


CONTEXT_DIM = 21
LATENT_DIM = 32
NUDGE_TYPES = ["N1", "N2", "N3", "N4", "N5", "N6"]


# ---------------------------------------------------------------------------
# Models — mirror app/backbones/centralized.py:68-91 exactly.
# ---------------------------------------------------------------------------
class BackboneEncoder(nn.Module):
    def __init__(self, input_dim: int = CONTEXT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.backbone(x)


class RewardPredictor(nn.Module):
    def __init__(self, input_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_interactions(path: Path) -> list[dict]:
    """
    Load interactions from a CSV/JSON file, or from a directory of such
    files (each file treated as one client; all rows concatenated into a
    single training pool). Each loaded interaction is tagged with
    ``_source`` = the originating filename for downstream diagnostics.
    """
    if path.is_dir():
        return _load_directory(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _tag_source(_load_csv(path), path.name)
    if suffix == ".json":
        return _tag_source(_load_json(path), path.name)
    raise ValueError(
        f"Unsupported path: {path} (expected .csv, .json, or a directory of them)"
    )


def _tag_source(rows: list[dict], source: str) -> list[dict]:
    for r in rows:
        r.setdefault("_source", source)
    return rows


def _load_directory(path: Path) -> list[dict]:
    """Concatenate every .csv and .json file in ``path`` into one list."""
    files = sorted(list(path.glob("*.csv")) + list(path.glob("*.json")))
    if not files:
        raise ValueError(f"No .csv or .json files found in directory {path}")
    all_rows: list[dict] = []
    for f in files:
        rows = _load_csv(f) if f.suffix.lower() == ".csv" else _load_json(f)
        rows = _tag_source(rows, f.name)
        stats = reward_baseline(rows)
        print(
            f"  loaded {len(rows):>5} from {f.name:<40} "
            f"reward mean={stats['mean']:+.3f} std={stats['std']:.3f} "
            f"range=[{stats['min']:+.3f}, {stats['max']:+.3f}]"
        )
        all_rows.extend(rows)
    return all_rows


def _load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # context column is a stringified Python list like "[0.5, -0.5, ...]"
            ctx = ast.literal_eval(row["context"])
            if len(ctx) != CONTEXT_DIM:
                raise ValueError(
                    f"Expected {CONTEXT_DIM}-dim context, got {len(ctx)} in {path}"
                )
            rows.append({"context": ctx, "reward": float(row["reward"])})
    return rows


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON interactions file must contain a list of dicts")
    for r in data:
        if "context" not in r or "reward" not in r:
            raise ValueError("Each interaction must have 'context' and 'reward' keys")
        if len(r["context"]) != CONTEXT_DIM:
            raise ValueError(
                f"Expected {CONTEXT_DIM}-dim context, got {len(r['context'])}"
            )
    return data


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _sample_context(rng: np.random.Generator) -> np.ndarray:
    """
    Sample one 21-dim context loosely matching the real CSV distribution.
    Nudge one-hot (dims 11-16) is set by the caller after picking a nudge.
    Behavioral features (dims 18-20) are set by generate_synthetic after this call.
    """
    ctx = np.zeros(CONTEXT_DIM, dtype=np.float64)
    # co2_reduction_rel: roughly uniform on [0.05, 0.9] in the real data
    ctx[0] = rng.uniform(0.02, 0.95)
    # co2_delta: in the real data, dim 1 ~= -dim 0 most of the time
    ctx[1] = -ctx[0]
    # price_delta_rel: often nonzero, sometimes 0
    ctx[2] = rng.uniform(-0.8, 0.3) if rng.random() < 0.75 else 0.0
    # calorie_delta and protein_delta: sometimes 0, sometimes spread
    ctx[3] = rng.uniform(-0.9, 1.0) if rng.random() < 0.65 else 0.0
    ctx[4] = rng.uniform(-0.95, 1.0) if rng.random() < 0.65 else 0.0
    # candidate_is_plant/meat/dairy (dims 5-7): essentially zero in real data
    # same_category_flag (dim 8): always 1 in real data
    ctx[8] = 1.0
    # similarity_score (dim 9): roughly [0.45, 0.96]
    ctx[9] = rng.uniform(0.45, 0.96)
    # is_lower_co2_flag (dim 10): 1 in real data
    ctx[10] = 1.0
    # cart_size_norm (dim 17): zero in real data
    return ctx


def generate_synthetic(
    n_clients: int,
    interactions_per_client: int,
    seed: int,
) -> list[dict]:
    """
    Linear-per-client preference model. Each client has weights w_c
    over the 21-dim context; raw_reward = w_c . ctx + bias + noise.
    The sign of raw_reward determines accept vs dismiss; accept rewards
    are mapped into [0.3, 0.95] to match observed accept rewards.
    Dims 18-20 are behavioral features computed from the client's actual
    interaction history up to (but not including) the current interaction,
    matching exactly what the real client does at inference time:
      18 — user_accept_rate:   running accept fraction over all history
      19 — recent_accept_rate: accept fraction over the last 10 interactions
      20 — interaction_count_norm: log-scaled interaction count
    Cold-start value for dims 18-19 when no history exists yet is 0.5.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    tuples: list[dict] = []

    for client_idx in range(n_clients):
        client_id = f"sim_{client_idx:04d}"
        w = rng.normal(0.0, 0.3, size=CONTEXT_DIM)
        bias = rng.normal(0.0, 0.1)

        accepted_count = 0
        recent_history: list[bool] = []  # last 10 outcomes, oldest first

        for i_interaction in range(interactions_per_client):
            ctx = _sample_context(rng)
            nudge = py_rng.choice(NUDGE_TYPES)
            # Set nudge one-hot (dims 11-16)
            for i, n in enumerate(NUDGE_TYPES):
                ctx[11 + i] = 1.0 if n == nudge else 0.0

            # Behavioral features from history *before* this interaction.
            ctx[18] = accepted_count / i_interaction if i_interaction > 0 else 0.5  # user_accept_rate
            recent = recent_history[-10:]
            ctx[19] = sum(recent) / len(recent) if recent else 0.5                  # recent_accept_rate
            ctx[20] = math.log(1 + i_interaction) / math.log(1 + 500)              # interaction_count_norm

            true_score = float(np.dot(w, ctx) + bias)
            raw_reward = true_score + float(rng.normal(0.0, 0.05))

            if raw_reward > 0:
                reward = float(min(0.95, max(0.30, 0.40 + 0.55 * math.tanh(raw_reward))))
                action = "accept"
            else:
                reward = -0.30
                action = "dismiss"

            accepted_count += raw_reward > 0
            recent_history.append(raw_reward > 0)

            tuples.append({
                "context": ctx.tolist(),
                "reward": reward,
                "client_id": client_id,
                "nudge_type": nudge,
                "action": action,
            })

    py_rng.shuffle(tuples)
    return tuples


# ---------------------------------------------------------------------------
# Reward baseline (irreducible MSE floor for a constant predictor)
# ---------------------------------------------------------------------------
def reward_baseline(interactions: list[dict]) -> dict:
    rewards = np.asarray([t["reward"] for t in interactions], dtype=np.float64)
    # Binary accept/reject label and its entropy. ``entropy_floor`` is the
    # BCE loss a model achieves by always predicting ``accept_rate`` —
    # the irreducible floor for binary classification, analogous to
    # ``var`` being the irreducible MSE floor.
    accept = (rewards > 0).astype(np.float64)
    p = float(accept.mean())
    if 0.0 < p < 1.0:
        entropy_floor = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    else:
        entropy_floor = 0.0
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "var": float(rewards.var()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "n": int(rewards.size),
        "accept_rate": p,
        "entropy_floor": entropy_floor,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _split_decay_params(module: nn.Module) -> tuple[list, list]:
    """Return (weight_params, bias_params). Used to keep bias out of weight_decay."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    return decay, no_decay


def train_one_run(
    interactions: list[dict],
    *,
    lr_backbone: float,
    lr_predictor: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    grad_clip: float,
    val_split: float,
    seed: int,
    objective: str = "mse",
) -> dict:
    """
    Train backbone + predictor on ``interactions``. ``objective`` selects
    the training signal:

    - ``mse``: predictor output is a scalar reward estimate; targets are
      z-scored raw rewards. The predictor's affine rescale is absorbed
      back into its weights at the end so it externally outputs raw
      rewards. This is the current production path.
    - ``bce``: predictor output is reinterpreted as a logit for
      ``P(accept) = P(reward > 0)``. Targets are the binary label
      ``reward > 0``. The bimodal real-data reward (dismiss=-0.3 vs
      accept in [0.3, 0.97]) has a hidden binary label that is much
      easier to learn than the noisy scalar regression target.

    Both objectives report MSE/MAE *and* BCE/accuracy each epoch so the
    two paths are directly comparable in ``compare`` output.
    """
    if objective not in ("mse", "bce"):
        raise ValueError(f"objective must be 'mse' or 'bce', got {objective!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    backbone = BackboneEncoder()
    predictor = RewardPredictor()

    # Bias is never weight-decayed: pulling it toward 0 fights its job of
    # absorbing the mean target (raw reward under MSE, logit prior under BCE).
    bb_decay, bb_no_decay = _split_decay_params(backbone)
    pred_decay, pred_no_decay = _split_decay_params(predictor)
    optimizer = optim.Adam(
        [
            {"params": bb_decay,      "lr": lr_backbone,  "weight_decay": weight_decay},
            {"params": bb_no_decay,   "lr": lr_backbone,  "weight_decay": 0.0},
            {"params": pred_decay,    "lr": lr_predictor, "weight_decay": weight_decay},
            {"params": pred_no_decay, "lr": lr_predictor, "weight_decay": 0.0},
        ],
    )

    contexts_all = torch.tensor(
        [t["context"] for t in interactions], dtype=torch.float32
    )
    rewards_all = torch.tensor(
        [t["reward"] for t in interactions], dtype=torch.float32
    ).unsqueeze(1)
    labels_all = (rewards_all > 0).float()

    py_rng = random.Random(seed)
    indices = list(range(len(interactions)))
    py_rng.shuffle(indices)
    n_val = int(len(indices) * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = contexts_all[train_idx]
    y_train_raw = rewards_all[train_idx]
    y_train_lbl = labels_all[train_idx]
    x_val = contexts_all[val_idx]
    y_val_raw = rewards_all[val_idx]
    y_val_lbl = labels_all[val_idx]

    # MSE-only: z-score target standardization, absorbed back at the end.
    # BCE targets are already in {0, 1} — no normalization needed.
    if objective == "mse":
        reward_mean = float(y_train_raw.mean().item())
        reward_std = float(y_train_raw.std(unbiased=False).item())
        if reward_std < 1e-6:
            reward_std = 1.0
        y_train_z = (y_train_raw - reward_mean) / reward_std
        with torch.no_grad():
            for name, p in predictor.named_parameters():
                if name.endswith("weight"):
                    p.div_(reward_std)
                elif name.endswith("bias"):
                    p.sub_(reward_mean).div_(reward_std)
    else:
        reward_mean = 0.0
        reward_std = 1.0
        y_train_z = y_train_raw  # unused under bce, kept for symmetry

    bce_loss_fn = nn.BCEWithLogitsLoss()

    params = [p for group in optimizer.param_groups for p in group["params"]]

    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        backbone.train()
        predictor.train()

        epoch_indices = list(range(len(x_train)))
        py_rng.shuffle(epoch_indices)

        # Accumulators serve both objectives so the history schema is
        # identical across runs. ``primary_loss_sum`` is the loss actually
        # backpropagated; the other (mse_sum_raw / bce_sum / abs_err_sum /
        # correct_count) is computed under no_grad for reporting only.
        primary_loss_sum = 0.0
        mse_sum_raw = 0.0
        bce_sum = 0.0
        abs_err_sum = 0.0
        correct_count = 0
        epoch_grad_sum = 0.0
        n_batches = 0
        n_seen = 0

        for start in range(0, len(epoch_indices), batch_size):
            batch_idx = epoch_indices[start : start + batch_size]
            xb = x_train[batch_idx]
            yb_raw = y_train_raw[batch_idx]
            yb_lbl = y_train_lbl[batch_idx]
            yb_z = y_train_z[batch_idx]

            emb = backbone(xb)
            pred = predictor(emb)  # interpreted as raw-reward (mse) or logit (bce)

            if objective == "mse":
                loss = nn.functional.mse_loss(pred, yb_z)
            else:
                loss = bce_loss_fn(pred, yb_lbl)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                if objective == "mse":
                    # ``pred`` is currently in z-score space; bring back to raw.
                    pred_raw = pred * reward_std + reward_mean
                    pred_logit = pred_raw  # treat raw-reward output as a (poor) logit
                else:
                    pred_logit = pred
                    pred_raw = torch.sigmoid(pred_logit)  # probability ~ "soft reward"

                mse_sum_raw += float(((pred_raw - yb_raw) ** 2).sum().item())
                abs_err_sum += float((pred_raw - yb_raw).abs().sum().item())
                bce_sum += float(
                    nn.functional.binary_cross_entropy_with_logits(
                        pred_logit, yb_lbl, reduction="sum"
                    ).item()
                )
                correct_count += int(((pred_logit > 0).float() == yb_lbl).sum().item())

            scale = (reward_std ** 2) if objective == "mse" else 1.0
            primary_loss_sum += float(loss.item()) * scale * len(batch_idx)
            epoch_grad_sum += float(grad_norm)
            n_batches += 1
            n_seen += len(batch_idx)

        train_loss = primary_loss_sum / max(n_seen, 1)
        train_mse = mse_sum_raw / max(n_seen, 1)
        train_bce = bce_sum / max(n_seen, 1)
        train_mae = abs_err_sum / max(n_seen, 1)
        train_acc = correct_count / max(n_seen, 1)
        avg_grad = epoch_grad_sum / max(n_batches, 1)

        backbone.eval()
        predictor.eval()
        with torch.no_grad():
            emb_train = backbone(x_train)
            emb_norm = float(emb_train.norm(dim=1).mean().item())
            if len(val_idx) > 0:
                pred_val = predictor(backbone(x_val))
                if objective == "mse":
                    pred_val_raw = pred_val * reward_std + reward_mean
                    pred_val_logit = pred_val_raw
                else:
                    pred_val_logit = pred_val
                    pred_val_raw = torch.sigmoid(pred_val_logit)

                val_mse = float(nn.functional.mse_loss(pred_val_raw, y_val_raw).item())
                val_mae = float((pred_val_raw - y_val_raw).abs().mean().item())
                val_bce = float(
                    nn.functional.binary_cross_entropy_with_logits(
                        pred_val_logit, y_val_lbl
                    ).item()
                )
                val_acc = float(((pred_val_logit > 0).float() == y_val_lbl).float().mean().item())
                val_loss = val_mse if objective == "mse" else val_bce
            else:
                val_mse = val_mae = val_bce = val_acc = val_loss = float("nan")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_bce": train_bce,
            "val_bce": val_bce,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "emb_norm": emb_norm,
            "grad_norm": avg_grad,
        })

        if objective == "mse":
            print(
                f"  epoch {epoch:3d}/{epochs}: "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} "
                f"val_acc={val_acc:.3f} emb_norm={emb_norm:.3f} grad_norm={avg_grad:.3f}"
            )
        else:
            print(
                f"  epoch {epoch:3d}/{epochs}: "
                f"train_bce={train_loss:.5f} val_bce={val_loss:.5f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                f"val_mse={val_mse:.5f} emb_norm={emb_norm:.3f} grad_norm={avg_grad:.3f}"
            )

    # MSE-only: absorb the z-score → raw-reward rescale back into the
    # predictor weights so the trained model outputs raw rewards externally.
    if objective == "mse":
        with torch.no_grad():
            for name, p in predictor.named_parameters():
                if name.endswith("weight"):
                    p.mul_(reward_std)
                elif name.endswith("bias"):
                    p.mul_(reward_std).add_(reward_mean)

    elapsed = time.perf_counter() - started
    final = history[-1] if history else {}
    best_val = (
        min(history, key=lambda r: r["val_loss"])
        if history and not math.isnan(history[0]["val_loss"])
        else {}
    )
    return {
        "objective": objective,
        "history": history,
        "final": final,
        "best_val": best_val,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_total": len(interactions),
        "training_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    tuples = generate_synthetic(
        n_clients=args.n_clients,
        interactions_per_client=args.interactions_per_client,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(tuples, f)
    stats = reward_baseline(tuples)
    print(f"Wrote {len(tuples)} interactions to {out}")
    print(
        f"  reward: mean={stats['mean']:.3f} std={stats['std']:.3f} "
        f"range=[{stats['min']:.3f}, {stats['max']:.3f}] var={stats['var']:.4f}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if data_path.is_dir():
        print(f"Loading interactions from directory {data_path}:")
    interactions = load_interactions(data_path)
    print(f"Loaded {len(interactions)} interactions total from {args.data}")
    stats = reward_baseline(interactions)
    print(
        f"  reward: mean={stats['mean']:.3f} std={stats['std']:.3f} "
        f"range=[{stats['min']:.3f}, {stats['max']:.3f}] var={stats['var']:.4f}"
    )
    print(
        f"  accept_rate={stats['accept_rate']:.3f} "
        f"(MSE floor = Var(reward) = {stats['var']:.4f}, "
        f"BCE floor = H(accept) = {stats['entropy_floor']:.4f})"
    )

    config = {
        "lr_backbone": args.lr_backbone,
        "lr_predictor": args.lr_predictor,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "val_split": args.val_split,
        "seed": args.seed,
        "objective": args.objective,
    }
    print("Hyperparameters:")
    for k, v in config.items():
        print(f"  {k} = {v}")

    result = train_one_run(interactions, **config)

    run_name = args.run_name or f"run_{int(time.time())}"
    out = (
        Path(args.out)
        if args.out
        else Path("results") / f"{run_name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_name": run_name,
        "config": {**config, "data_file": str(Path(args.data).resolve())},
        "reward_stats": stats,
        **result,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print(
        f"Final: train_loss={result['final'].get('train_loss', float('nan')):.5f} "
        f"val_loss={result['final'].get('val_loss', float('nan')):.5f}"
    )
    if result["best_val"]:
        bv = result["best_val"]
        print(f"Best val_loss={bv['val_loss']:.5f} at epoch {bv['epoch']}")
    print(f"Wrote results to {out}")


def cmd_compare(args: argparse.Namespace) -> None:
    rows = []
    for path_str in args.runs:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        final = data.get("final", {})
        best = data.get("best_val", {})
        reward_stats = data.get("reward_stats", {})
        obj = data.get("objective") or cfg.get("objective") or "mse"
        # Each objective's natural floor: MSE→Var(reward), BCE→H(accept).
        floor = (
            reward_stats.get("var", float("nan"))
            if obj == "mse"
            else reward_stats.get("entropy_floor", float("nan"))
        )
        rows.append({
            "name": data.get("run_name", path.stem),
            "obj": obj,
            "lr_bb": cfg.get("lr_backbone"),
            "lr_pred": cfg.get("lr_predictor"),
            "epochs": cfg.get("epochs"),
            "bs": cfg.get("batch_size"),
            "wd": cfg.get("weight_decay"),
            "final_train": final.get("train_loss"),
            "final_val": final.get("val_loss"),
            "val_acc": final.get("val_acc"),
            "best_val": best.get("val_loss"),
            "best_epoch": best.get("epoch"),
            "floor": floor,
        })

    cols = [
        ("name", 22),
        ("obj", 4),
        ("lr_bb", 8),
        ("lr_pred", 8),
        ("epochs", 6),
        ("bs", 4),
        ("wd", 8),
        ("final_train", 11),
        ("final_val", 10),
        ("val_acc", 8),
        ("best_val", 10),
        ("best_epoch", 10),
        ("floor", 10),
    ]
    header = " ".join(f"{name:>{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for col, w in cols:
            v = row[col]
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = f"{v:.5f}" if abs(v) < 100 else f"{v:.2e}"
            else:
                s = str(v)
            cells.append(f"{s:>{w}}")
        print(" ".join(cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone trainer for the centralized backbone + reward predictor.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic interaction data.")
    g.add_argument("--n-clients", type=int, default=50)
    g.add_argument("--interactions-per-client", type=int, default=100)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", required=True, help="Output JSON path.")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Train one model run and save a results JSON.")
    t.add_argument("--data", required=True,
                   help="Path to a CSV/JSON interactions file, or a directory "
                        "of them (each file = one client, all concatenated).")
    t.add_argument("--lr-backbone", type=float, default=3e-4)
    t.add_argument("--lr-predictor", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--val-split", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument(
        "--objective", choices=["mse", "bce"], default="mse",
        help="Training loss: 'mse' (current) or 'bce' "
             "(binary accept/reject on reward > 0).",
    )
    t.add_argument("--run-name", default=None,
                   help="Tag for this run; default = run_<unix-timestamp>.")
    t.add_argument("--out", default=None,
                   help="Output results JSON path (default: results/<run-name>.json).")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="Compare multiple results JSONs in a table.")
    c.add_argument("runs", nargs="+", help="Paths to results JSON files.")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
