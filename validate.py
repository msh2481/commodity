#!/usr/bin/env python3
"""Kick off a validation run: queue K fold-jobs to pueue, wait for them,
aggregate scores, and append a row to validation/log.csv.

Usage:
    ./validate "short description of this run" path/to/model.py

`path/to/model.py` must expose a class `Model` with:
    class Model:
        device = "cpu"            # or "cuda"
        cpus_per_fold = None      # None -> auto-split N_CPUS / K

        def __init__(self): ...
        def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None: ...
        def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
VAL_DIR = REPO_ROOT / "validation"
FOLDS_CSV = VAL_DIR / "fold_assignments.csv"
LOG_CSV = VAL_DIR / "log.csv"
PREDS_CACHE = VAL_DIR / "predictions_cache"
MODELS_CACHE = VAL_DIR / "models_cache"
RUNS_DIR = VAL_DIR / "runs"
FOLD_WORKER = VAL_DIR / "fold_worker.py"

N_FOLDS = 4


def detect_hardware() -> tuple[int, int]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n_gpus = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (FileNotFoundError, subprocess.CalledProcessError):
        n_gpus = 0
    n_cpus = os.cpu_count() or 1
    return n_gpus, n_cpus


def ensure_folds(n_folds: int) -> pd.DataFrame:
    if FOLDS_CSV.exists():
        folds = pd.read_csv(FOLDS_CSV)
        if folds.fold.nunique() != n_folds:
            raise RuntimeError(
                f"{FOLDS_CSV} has {folds.fold.nunique()} folds, expected {n_folds}. "
                "Delete the file to regenerate."
            )
        return folds

    labels = pd.read_csv(DATA_DIR / "train_labels.csv", usecols=["date_id"])
    date_ids = labels.date_id.sort_values().to_numpy()
    assignments = np.repeat(np.arange(n_folds), len(date_ids) // n_folds + 1)[: len(date_ids)]
    folds = pd.DataFrame({"date_id": date_ids, "fold": assignments})
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"[validate] wrote {FOLDS_CSV} ({len(folds)} dates, {n_folds} folds)", flush=True)
    return folds


def ensure_pueued() -> None:
    """Start pueued in the background if it's not already running."""
    r = subprocess.run(["pueue", "status"], capture_output=True, text=True)
    if r.returncode == 0:
        return
    print("[validate] starting pueued in background", flush=True)
    subprocess.Popen(
        ["pueued", "-d"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.2)
        r = subprocess.run(["pueue", "status"], capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError("pueued failed to start (pueue status still fails)")


def ensure_pueue_group(name: str, parallel: int) -> None:
    r = subprocess.run(["pueue", "group", "-j"], capture_output=True, text=True, check=True)
    groups = json.loads(r.stdout)
    if name not in groups:
        subprocess.run(["pueue", "group", "add", name, "--parallel", str(parallel)], check=True)
    else:
        current = groups[name].get("parallel_tasks")
        if current != parallel:
            subprocess.run(
                ["pueue", "parallel", "--group", name, str(parallel)], check=True
            )


def query_model(model_path: Path) -> dict:
    """Import the user's model.py in a throwaway subprocess to read its
    class-level attributes without initializing CUDA in our parent process.
    """
    probe = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{model_path}')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "M = mod.Model\n"
        "print(json.dumps({"
        "'device': getattr(M, 'device', 'cpu'),"
        "'cpus_per_fold': getattr(M, 'cpus_per_fold', None)"
        "}))\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # block CUDA in the probe
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"failed to query model attributes from {model_path}:\n{r.stderr}"
        )
    return json.loads(r.stdout.strip().splitlines()[-1])


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s or "run")[:maxlen]


def queue_fold(
    fold_idx: int,
    group: str,
    env_prefix: list[str],
    model_path: Path,
    out_path: Path,
    label: str,
) -> int:
    cmd = env_prefix + [
        sys.executable,
        str(FOLD_WORKER),
        "--fold", str(fold_idx),
        "--model", str(model_path),
        "--data-dir", str(DATA_DIR),
        "--folds", str(FOLDS_CSV),
        "--out", str(out_path),
    ]
    r = subprocess.run(
        [
            "pueue", "add",
            "--group", group,
            "--working-directory", str(REPO_ROOT),
            "--label", label,
            "--print-task-id",
            "--", *cmd,
        ],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


def wait_for_tasks(task_ids: list[int]) -> None:
    subprocess.run(["pueue", "wait", *map(str, task_ids)], check=True)


def task_statuses(task_ids: list[int]) -> dict[int, dict]:
    r = subprocess.run(
        ["pueue", "status", "-j"],
        capture_output=True, text=True, check=True,
    )
    tasks = json.loads(r.stdout)["tasks"]
    return {tid: tasks[str(tid)] for tid in task_ids if str(tid) in tasks}


def task_succeeded(status: dict) -> bool:
    s = status.get("status")
    if not isinstance(s, dict) or "Done" not in s:
        return False
    result = s["Done"].get("result")
    return result == "Success"


def _parse_ts(s: str) -> datetime:
    """Parse pueue's RFC3339 timestamp. Handles nanosecond precision by
    truncating to microseconds."""
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        frac = frac[:6].ljust(6, "0")
        s = f"{head}.{frac}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def task_runtime_seconds(status: dict) -> float | None:
    """Actual run time (start -> end) of a pueue task, excluding queue wait."""
    s = status.get("status", {}).get("Done", {})
    if not s or "start" not in s or "end" not in s:
        return None
    return (_parse_ts(s["end"]) - _parse_ts(s["start"])).total_seconds()


def summarize_failure(task_id: int) -> str:
    r = subprocess.run(
        ["pueue", "log", "--lines", "80", "-j", str(task_id)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(r.stdout)
        entry = next(iter(data.values()))
        out = entry.get("output", "")
        return out[-3000:]
    except Exception:
        return r.stdout[-3000:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", help="human-readable description of this run")
    parser.add_argument("model", type=Path, help="path to model.py")
    args = parser.parse_args()

    model_arg = args.model
    if not model_arg.is_absolute():
        orig_pwd = os.environ.get("VALIDATE_ORIG_PWD")
        if orig_pwd:
            alt = Path(orig_pwd) / model_arg
            if alt.exists():
                model_arg = alt
    model_src = model_arg.resolve()
    if not model_src.exists():
        print(f"model file not found: {model_src}", file=sys.stderr)
        return 2

    for d in (VAL_DIR, PREDS_CACHE, MODELS_CACHE, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    n_gpus, n_cpus = detect_hardware()
    print(f"[validate] detected {n_gpus} GPUs, {n_cpus} CPUs", flush=True)

    folds = ensure_folds(N_FOLDS)

    model_info = query_model(model_src)
    device = model_info["device"]
    if device not in ("cpu", "cuda"):
        raise RuntimeError(f"Model.device must be 'cpu' or 'cuda', got {device!r}")

    if model_info["cpus_per_fold"] is not None:
        cpus_per_fold = int(model_info["cpus_per_fold"])
    else:
        cpus_per_fold = max(1, n_cpus // N_FOLDS)
    print(
        f"[validate] model.device={device}  cpus_per_fold={cpus_per_fold}",
        flush=True,
    )

    ensure_pueued()
    if n_gpus > 0:
        for i in range(n_gpus):
            ensure_pueue_group(f"gpu{i}", parallel=1)
    ensure_pueue_group("cpu", parallel=N_FOLDS)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{ts}_{slugify(args.description)}"
    run_dir = RUNS_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the model file early so it's preserved even if the run fails.
    model_snapshot = MODELS_CACHE / f"{slug}.py"
    shutil.copy2(model_src, model_snapshot)

    task_ids: list[int] = []
    for fold_idx in range(N_FOLDS):
        if device == "cuda":
            if n_gpus == 0:
                raise RuntimeError("model.device='cuda' but no GPUs detected")
            gpu_id = fold_idx % n_gpus
            group = f"gpu{gpu_id}"
            env_prefix = [
                "env",
                f"CUDA_VISIBLE_DEVICES={gpu_id}",
                f"OMP_NUM_THREADS={cpus_per_fold}",
                f"MKL_NUM_THREADS={cpus_per_fold}",
                f"OPENBLAS_NUM_THREADS={cpus_per_fold}",
                f"NUMEXPR_NUM_THREADS={cpus_per_fold}",
                "PYTHONUNBUFFERED=1",
            ]
        else:
            group = "cpu"
            env_prefix = [
                "env",
                "CUDA_VISIBLE_DEVICES=",
                f"OMP_NUM_THREADS={cpus_per_fold}",
                f"MKL_NUM_THREADS={cpus_per_fold}",
                f"OPENBLAS_NUM_THREADS={cpus_per_fold}",
                f"NUMEXPR_NUM_THREADS={cpus_per_fold}",
                "PYTHONUNBUFFERED=1",
            ]
        out_path = run_dir / f"fold_{fold_idx}.csv"
        label = f"validate:{slug}:fold{fold_idx}"
        task_id = queue_fold(
            fold_idx, group, env_prefix, model_src, out_path, label
        )
        task_ids.append(task_id)
        print(f"[validate] queued fold {fold_idx} -> group {group} task {task_id}", flush=True)

    print(f"[validate] waiting for tasks {task_ids}", flush=True)
    wait_for_tasks(task_ids)
    statuses = task_statuses(task_ids)
    failed = [tid for tid in task_ids if not task_succeeded(statuses[tid])]
    fold_runtimes = {tid: task_runtime_seconds(statuses[tid]) for tid in task_ids}
    if failed:
        for tid in failed:
            print(f"[validate] fold task {tid} failed:", file=sys.stderr)
            print(summarize_failure(tid), file=sys.stderr)
        return 1

    from validation.scoring import per_date_spearman, sharpe_from_rhos

    y_full = pd.read_csv(DATA_DIR / "train_labels.csv").set_index("date_id")

    fold_metrics = []
    fold_preds_parts = []
    for fold_idx in range(N_FOLDS):
        out_path = run_dir / f"fold_{fold_idx}.csv"
        preds = pd.read_csv(out_path).set_index("date_id")
        fold_preds_parts.append(preds)
        truth = y_full.loc[preds.index.intersection(y_full.index)]
        rhos = per_date_spearman(preds.loc[truth.index], truth)
        sharpe = sharpe_from_rhos(rhos)
        # optional train predictions (if the fold_worker wrote them)
        train_path = out_path.with_name(out_path.stem + "_train.csv")
        train_sharpe = float("nan")
        train_mean_rho = float("nan")
        if train_path.exists():
            tp = pd.read_csv(train_path).set_index("date_id")
            t_truth = y_full.loc[tp.index.intersection(y_full.index)]
            if len(t_truth):
                t_rhos = per_date_spearman(tp.loc[t_truth.index], t_truth)
                train_sharpe = sharpe_from_rhos(t_rhos)
                train_mean_rho = float(t_rhos.mean()) if t_rhos.notna().any() else float("nan")
        fold_metrics.append({
            "fold": fold_idx,
            "n": int(rhos.notna().sum()),
            "mean_rho": float(rhos.mean()) if rhos.notna().any() else float("nan"),
            "std_rho": float(rhos.std(ddof=1)) if rhos.notna().sum() >= 2 else float("nan"),
            "sharpe": sharpe,
            "train_sharpe": train_sharpe,
            "train_mean_rho": train_mean_rho,
            "runtime_seconds": fold_runtimes[task_ids[fold_idx]],
        })

    runtimes = [fm["runtime_seconds"] for fm in fold_metrics if fm["runtime_seconds"] is not None]
    eval_time_seconds = float(sum(runtimes)) if runtimes else float("nan")
    eval_wallclock_seconds = float(max(runtimes)) if runtimes else float("nan")

    all_preds = pd.concat(fold_preds_parts).sort_index().copy()
    all_truth = y_full.loc[all_preds.index.intersection(y_full.index)]
    overall_rhos = per_date_spearman(all_preds.loc[all_truth.index], all_truth)
    overall = {
        "n": int(overall_rhos.notna().sum()),
        "mean_rho": float(overall_rhos.mean()) if overall_rhos.notna().any() else float("nan"),
        "std_rho": float(overall_rhos.std(ddof=1)) if overall_rhos.notna().sum() >= 2 else float("nan"),
        "sharpe": sharpe_from_rhos(overall_rhos),
    }

    preds_cache = PREDS_CACHE / f"{slug}.csv"
    all_preds.reset_index().to_csv(preds_cache, index=False)

    log_row = {
        "timestamp": ts,
        "slug": slug,
        "description": args.description,
        "model_src": str(model_src),
        "model_cache": str(model_snapshot),
        "preds_cache": str(preds_cache),
        "overall_sharpe": overall["sharpe"],
        "overall_mean_rho": overall["mean_rho"],
        "overall_std_rho": overall["std_rho"],
        "overall_n": overall["n"],
        "eval_time_seconds": eval_time_seconds,
        "eval_wallclock_seconds": eval_wallclock_seconds,
    }
    for fm in fold_metrics:
        i = fm["fold"]
        log_row[f"fold_{i}_sharpe"] = fm["sharpe"]
        log_row[f"fold_{i}_mean_rho"] = fm["mean_rho"]
        log_row[f"fold_{i}_std_rho"] = fm["std_rho"]
        log_row[f"fold_{i}_n"] = fm["n"]
        log_row[f"fold_{i}_runtime_seconds"] = fm["runtime_seconds"]
        log_row[f"fold_{i}_train_sharpe"] = fm["train_sharpe"]
        log_row[f"fold_{i}_train_mean_rho"] = fm["train_mean_rho"]

    is_new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(log_row)

    # Human-readable summary
    print()
    print(f"=== {slug} ===")
    print(f"description:  {args.description}")
    print(f"overall:      Sharpe={overall['sharpe']:+.4f}  mean_rho={overall['mean_rho']:+.5f}  std_rho={overall['std_rho']:.5f}  n={overall['n']}")
    for fm in fold_metrics:
        rt = fm["runtime_seconds"]
        rt_str = f"{rt:6.1f}s" if rt is not None else "   ?   "
        ts_str = (
            f"train_Sharpe={fm['train_sharpe']:+.4f}"
            if not np.isnan(fm.get("train_sharpe", float("nan")))
            else "train_Sharpe=NA"
        )
        print(
            f"fold {fm['fold']}:       Sharpe={fm['sharpe']:+.4f}  {ts_str}  "
            f"mean_rho={fm['mean_rho']:+.5f}  std_rho={fm['std_rho']:.5f}  n={fm['n']}  runtime={rt_str}"
        )
    others = [fm["sharpe"] for fm in fold_metrics[:-1] if not np.isnan(fm["sharpe"])]
    last = fold_metrics[-1]["sharpe"]
    if others and not np.isnan(last):
        delta = last - float(np.mean(others))
        print(f"last-fold-delta (fold{N_FOLDS-1} - mean(others)):  {delta:+.4f}")
    print(
        f"eval time:    compute={eval_time_seconds:.1f}s  "
        f"wallclock={eval_wallclock_seconds:.1f}s"
    )
    print(f"cached preds: {preds_cache}")
    print(f"log:          {LOG_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
