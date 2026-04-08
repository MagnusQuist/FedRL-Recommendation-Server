from __future__ import annotations

import asyncio
import itertools
import json
import logging
import shutil
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from pretrain.run import PretrainOptions, run_pretraining
from app.db import AsyncSessionLocal


class _BufferHandler(logging.Handler):
    def __init__(self, target: deque[str]) -> None:
        super().__init__()
        self._target = target
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._target.append(self.format(record))
        except Exception:
            self.handleError(record)


class JobState:
    def __init__(self) -> None:
        self.status: Literal["idle", "running", "success", "failed"] = "idle"
        self.job_type: Literal["single", "sweep"] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.last_error: str | None = None
        self.last_result: dict | None = None
        self.last_options: dict | None = None
        self.sweep_total: int = 0
        self.sweep_completed: int = 0
        self.sweep_top_runs: list[dict] = []
        self.logs: deque[str] = deque(maxlen=2000)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start(self, options: PretrainOptions) -> bool:
        async with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.started_at = _utc_now()
            self.finished_at = None
            self.last_error = None
            self.last_result = None
            self.job_type = "single"
            self.last_options = asdict(options)
            self.sweep_total = 0
            self.sweep_completed = 0
            self.sweep_top_runs = []
            self.logs.clear()
            self._task = asyncio.create_task(self._run(options))
            return True

    async def start_sweep(self, options_list: list[PretrainOptions], sweep_id: str) -> bool:
        async with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.job_type = "sweep"
            self.started_at = _utc_now()
            self.finished_at = None
            self.last_error = None
            self.last_result = None
            self.last_options = {"sweep_id": sweep_id}
            self.sweep_total = len(options_list)
            self.sweep_completed = 0
            self.sweep_top_runs = []
            self.logs.clear()
            self._task = asyncio.create_task(self._run_sweep(options_list, sweep_id))
            return True

    async def _run(self, options: PretrainOptions) -> None:
        log_handler = _BufferHandler(self.logs)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        old_level = root_logger.level
        if old_level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        try:
            self.logs.append("Starting training job...")
            self.last_result = await run_pretraining(options)
            self.status = "success"
            self.logs.append("Training job finished successfully.")
        except Exception as exc:
            self.status = "failed"
            self.last_error = str(exc)
            self.logs.append(f"Training job failed: {exc}")
        finally:
            self.finished_at = _utc_now()
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(old_level)

    async def _run_sweep(self, options_list: list[PretrainOptions], sweep_id: str) -> None:
        log_handler = _BufferHandler(self.logs)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        old_level = root_logger.level
        if old_level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        top_results: list[dict] = []
        try:
            self.logs.append(f"Starting sweep '{sweep_id}' with {len(options_list)} combinations.")
            for idx, options in enumerate(options_list, start=1):
                self.logs.append(
                    f"[{idx}/{len(options_list)}] epochs={options.epochs} lr={options.lr} "
                    f"batch={options.batch_size} patience={options.patience} seed={options.seed}"
                )
                try:
                    result = await run_pretraining(options)
                    score = float(result.get("best_val_loss", float("inf")))
                    top_results.append(
                        {
                            "score": score,
                            "run_id": Path(result.get("artifacts_dir", "")).name if result.get("artifacts_dir") else None,
                            "result": result,
                        }
                    )
                    top_results.sort(key=lambda item: item["score"])
                    if len(top_results) > 5:
                        removed = top_results.pop(-1)
                        removed_dir = removed.get("result", {}).get("artifacts_dir")
                        if removed_dir:
                            try:
                                shutil.rmtree(removed_dir, ignore_errors=True)
                            except Exception:
                                pass
                    self.sweep_top_runs = [
                        {"run_id": r["run_id"], "best_val_loss": r["score"]} for r in top_results
                    ]
                except Exception as exc:
                    self.logs.append(f"[{idx}/{len(options_list)}] FAILED: {exc}")
                finally:
                    self.sweep_completed = idx

            self.last_result = {
                "sweep_id": sweep_id,
                "total_candidates": len(options_list),
                "top_runs": self.sweep_top_runs,
            }
            self.status = "success"
            self.logs.append("Sweep completed.")
        except Exception as exc:
            self.status = "failed"
            self.last_error = str(exc)
            self.logs.append(f"Sweep failed: {exc}")
        finally:
            self.finished_at = _utc_now()
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(old_level)

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "job_type": self.job_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "last_result": self.last_result,
            "last_options": self.last_options,
            "sweep_total": self.sweep_total,
            "sweep_completed": self.sweep_completed,
            "sweep_top_runs": self.sweep_top_runs,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


app = FastAPI(title="FedRL Pretrain UI")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
state = JobState()
training_root = Path(__file__).parent / "training_data"
training_root.mkdir(parents=True, exist_ok=True)
app.mount("/training_data", StaticFiles(directory=str(training_root)), name="training_data")


def _make_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = training_root / f"run_{ts}"
    suffix = 1
    while run_dir.exists():
        run_dir = training_root / f"run_{ts}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _make_sweep_run_dir(sweep_id: str, trial_idx: int) -> Path:
    run_dir = training_root / f"{sweep_id}_trial_{trial_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _int_range(min_v: int, max_v: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("Integer step must be positive.")
    if min_v > max_v:
        raise ValueError("Min value must be <= max value.")
    values: list[int] = []
    curr = min_v
    while curr <= max_v:
        values.append(curr)
        curr += step
    return values


def _float_range(min_v: float, max_v: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Float step must be positive.")
    if min_v > max_v:
        raise ValueError("Min value must be <= max value.")
    values: list[float] = []
    curr = min_v
    while curr <= max_v + 1e-12:
        values.append(round(curr, 10))
        curr += step
    return values


def _list_runs() -> list[dict]:
    runs: list[dict] = []
    for run_dir in training_root.iterdir():
        if not run_dir.is_dir():
            continue
        is_single_run = run_dir.name.startswith("run_")
        is_sweep_trial = run_dir.name.startswith("sweep_")
        if not is_single_run and not is_sweep_trial:
            continue
        summary_path = run_dir / "summary.json"
        plot_path = run_dir / "training_loss.png"
        weights_path = run_dir / "backbone_weights.npz"
        summary_payload = None
        if summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_payload = None
        runs.append(
            {
                "id": run_dir.name,
                "kind": "sweep_trial" if is_sweep_trial else "single",
                "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat(),
                "has_summary": summary_path.exists(),
                "has_plot": plot_path.exists(),
                "has_weights": weights_path.exists(),
                "best_val_loss": (
                    summary_payload.get("result", {}).get("best_val_loss")
                    if isinstance(summary_payload, dict)
                    else None
                ),
            }
        )
    runs.sort(key=lambda item: item["id"], reverse=True)
    return runs


def _best_run(runs: list[dict]) -> dict | None:
    candidates = [r for r in runs if isinstance(r.get("best_val_loss"), (int, float))]
    if not candidates:
        return None
    winner = min(candidates, key=lambda r: float(r["best_val_loss"]))
    return dict(winner)


def _safe_run_path(run_id: str) -> Path | None:
    if (not run_id.startswith("run_") and not run_id.startswith("sweep_")) or "/" in run_id or "\\" in run_id:
        return None
    run_path = training_root / run_id
    if not run_path.exists() or not run_path.is_dir():
        return None
    return run_path


@app.on_event("startup")
async def startup_probe() -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        logging.getLogger(__name__).info("Database connectivity check passed.")
    except Exception as exc:
        logging.getLogger(__name__).warning("Database connectivity check failed at startup: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    runs = _list_runs()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"status": state.snapshot(), "runs": runs, "best_run": _best_run(runs)},
    )


@app.post("/train")
async def train(
    epochs: int = Form(300),
    lr: float = Form(1e-3),
    batch_size: int = Form(256),
    patience: int = Form(20),
    seed: int = Form(42),
    no_save: bool = Form(False),
) -> RedirectResponse:
    run_dir = _make_run_dir()
    options = PretrainOptions(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
        no_save=no_save,
        output_dir=str(run_dir),
    )
    await state.start(options)
    return RedirectResponse(url="/", status_code=303)


@app.post("/sweep")
async def sweep(
    epochs_min: int = Form(100),
    epochs_max: int = Form(300),
    epochs_step: int = Form(100),
    lr_min: float = Form(1e-4),
    lr_max: float = Form(1e-3),
    lr_step: float = Form(3e-4),
    batch_min: int = Form(128),
    batch_max: int = Form(256),
    batch_step: int = Form(128),
    patience_min: int = Form(10),
    patience_max: int = Form(20),
    patience_step: int = Form(10),
    seed_min: int = Form(42),
    seed_max: int = Form(42),
    seed_step: int = Form(1),
    no_save: bool = Form(True),
) -> RedirectResponse:
    try:
        epochs_values = _int_range(epochs_min, epochs_max, epochs_step)
        lr_values = _float_range(lr_min, lr_max, lr_step)
        batch_values = _int_range(batch_min, batch_max, batch_step)
        patience_values = _int_range(patience_min, patience_max, patience_step)
        seed_values = _int_range(seed_min, seed_max, seed_step)
    except ValueError as exc:
        state.last_error = str(exc)
        return RedirectResponse(url="/", status_code=303)

    combos = list(itertools.product(epochs_values, lr_values, batch_values, patience_values, seed_values))
    max_combos = 250
    if len(combos) > max_combos:
        state.last_error = f"Too many combinations ({len(combos)}). Reduce ranges to <= {max_combos}."
        return RedirectResponse(url="/", status_code=303)

    sweep_id = f"sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    options_list: list[PretrainOptions] = []
    for idx, (epochs, lr, batch, patience, seed) in enumerate(combos, start=1):
        run_dir = _make_sweep_run_dir(sweep_id, idx)
        options_list.append(
            PretrainOptions(
                epochs=epochs,
                lr=lr,
                batch_size=batch,
                patience=patience,
                seed=seed,
                no_save=no_save,
                output_dir=str(run_dir),
            )
        )

    await state.start_sweep(options_list, sweep_id=sweep_id)
    return RedirectResponse(url="/", status_code=303)


@app.get("/status")
async def get_status() -> JSONResponse:
    return JSONResponse(state.snapshot())


@app.get("/logs")
async def get_logs() -> JSONResponse:
    return JSONResponse({"logs": list(state.logs)})


@app.get("/runs")
async def get_runs() -> JSONResponse:
    runs = _list_runs()
    return JSONResponse({"runs": runs, "best_run": _best_run(runs)})


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    run_path = _safe_run_path(run_id)
    if run_path is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    summary_path = run_path / "summary.json"
    summary_payload = None
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary_payload = {"error": f"Could not parse summary.json: {exc}"}

    payload = {
        "id": run_id,
        "summary": summary_payload,
        "plot_url": f"/training_data/{run_id}/training_loss.png" if (run_path / "training_loss.png").exists() else None,
        "weights_url": f"/training_data/{run_id}/backbone_weights.npz" if (run_path / "backbone_weights.npz").exists() else None,
        "summary_url": f"/training_data/{run_id}/summary.json" if summary_path.exists() else None,
    }
    return JSONResponse(payload)

