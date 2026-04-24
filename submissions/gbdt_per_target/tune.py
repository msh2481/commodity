"""Skopt-driven hparam search for v38_tune.py.

Usage: python3 tune.py [--trials N] [--seed K] [--reset]

- Reads existing trials from _tune/trials.jsonl so restarts resume.
- First K trials (default 12): LHS seed grid. Then: GP + EIps.
- Each trial: write _tune/hparams.json → ./validate → parse stdout → tell skopt.
- Appends one line to _tune/trials.jsonl and one line to journal.jsonl per trial.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from skopt import Optimizer
    from skopt.space import Real, Integer
except ImportError:
    print("skopt not installed. pip install scikit-optimize", file=sys.stderr)
    sys.exit(1)


THREAD_DIR = Path("/root/commodity/submissions/gbdt_per_target")
TUNE_DIR = THREAD_DIR / "_tune"
TUNE_DIR.mkdir(exist_ok=True)
HPARAMS_FILE = TUNE_DIR / "hparams.json"
TRIALS_LOG = TUNE_DIR / "trials.jsonl"
JOURNAL = THREAD_DIR / "journal.jsonl"
MODEL_FILE = THREAD_DIR / "v38_tune.py"
VALIDATE = Path("/root/commodity/validate")
REPO_ROOT = Path("/root/commodity")


SPACE = [
    Integer(50, 500, name="n_estimators"),
    Integer(3, 8, name="max_depth"),
    Real(0.01, 0.2, prior="log-uniform", name="learning_rate"),
    Integer(1, 50, name="min_child_weight"),
    Real(0.5, 1.0, name="subsample"),
    Real(0.3, 1.0, name="colsample_bytree"),
    Real(0.01, 50.0, prior="log-uniform", name="reg_lambda"),
    Real(0.001, 10.0, prior="log-uniform", name="reg_alpha"),
    Real(1e-3, 1.0, prior="log-uniform", name="gamma"),
]
NAMES = [s.name for s in SPACE]


OVERALL_RE = re.compile(r"^overall:\s+Sharpe=([+-]?\d+\.\d+)", re.MULTILINE)
FOLD_RE = re.compile(r"^fold (\d+):\s+Sharpe=([+-]?\d+\.\d+).*?runtime=\s*([\d.]+)s", re.MULTILINE)
EVAL_TIME_RE = re.compile(r"eval time:\s+compute=([\d.]+)s\s+wallclock=([\d.]+)s")
SLUG_RE = re.compile(r"^=== (\S+) ===", re.MULTILINE)


def _cast_params(x):
    """Map skopt x-array to Python dict with native types (no numpy scalars)."""
    out = {}
    for name, val in zip(NAMES, x):
        if isinstance(val, (np.integer, np.int32, np.int64)):
            out[name] = int(val)
        elif isinstance(val, (np.floating, np.float32, np.float64)):
            out[name] = float(val)
        else:
            out[name] = val
    return out


def load_history():
    """Return list of (x_list, neg_sharpe, eval_s, sharpe, params_dict)."""
    if not TRIALS_LOG.exists():
        return []
    rows = []
    for line in TRIALS_LOG.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("status") != "done":
            continue
        p = d["params"]
        try:
            x = [p[k] for k in NAMES]
        except KeyError:
            continue
        sh = d.get("overall_sharpe")
        if sh is None or np.isnan(sh):
            continue
        ev = d.get("eval_time_seconds") or 60.0
        rows.append((x, -sh, ev, sh, p))
    return rows


def write_hparams(params):
    HPARAMS_FILE.write_text(json.dumps(params, indent=2))


def run_validate(trial_idx, params):
    desc = f"v38_tune trial {trial_idx:03d}: " + \
           f"ne={params['n_estimators']} md={params['max_depth']} lr={params['learning_rate']:.3f}"
    # Limit description to something readable
    cmd = [str(VALIDATE), desc, str(MODEL_FILE)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1200)
    wall = time.time() - t0
    return proc, wall


def parse_output(stdout):
    slug_m = SLUG_RE.search(stdout)
    overall_m = OVERALL_RE.search(stdout)
    eval_m = EVAL_TIME_RE.search(stdout)
    fold_matches = FOLD_RE.findall(stdout)

    slug = slug_m.group(1) if slug_m else None
    sharpe = float(overall_m.group(1)) if overall_m else float("nan")
    eval_time = float(eval_m.group(1)) if eval_m else None
    wallclock = float(eval_m.group(2)) if eval_m else None
    folds = [float(f[1]) for f in sorted(fold_matches, key=lambda x: int(x[0]))]
    return slug, sharpe, eval_time, wallclock, folds


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_trial(trial_idx, params, opt):
    t_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_hparams(params)
    proc, wall = run_validate(trial_idx, params)

    if proc.returncode != 0:
        row = {
            "timestamp": t_stamp, "trial": trial_idx, "params": params,
            "status": "failed",
            "stderr_tail": (proc.stderr or "")[-500:],
            "stdout_tail": (proc.stdout or "")[-500:],
        }
        append_jsonl(TRIALS_LOG, row)
        return None, None

    slug, sharpe, eval_time, wallclock, folds = parse_output(proc.stdout)
    if sharpe != sharpe:  # NaN
        row = {
            "timestamp": t_stamp, "trial": trial_idx, "params": params,
            "status": "failed", "note": "no sharpe parsed",
            "stdout_tail": proc.stdout[-800:],
        }
        append_jsonl(TRIALS_LOG, row)
        return None, None

    row = {
        "timestamp": t_stamp,
        "trial": trial_idx,
        "slug": slug,
        "params": params,
        "overall_sharpe": sharpe,
        "fold_sharpes": folds,
        "eval_time_seconds": eval_time,
        "eval_wallclock_seconds": wallclock,
        "status": "done",
    }
    append_jsonl(TRIALS_LOG, row)

    # Mirror to the thread journal (for cross-reference)
    journal_row = {
        "timestamp": t_stamp,
        "slug": slug,
        "attempt": f"v38_tune_t{trial_idx:03d}",
        "description": f"skopt trial {trial_idx}",
        "params": params,
        "fold_sharpes": folds,
        "overall_sharpe": sharpe,
        "eval_time_seconds": eval_time,
        "status": "done",
        "notes": f"tune trial {trial_idx}; features=v38 combined; mask=lag>=4 & has_LME",
    }
    append_jsonl(JOURNAL, journal_row)
    return sharpe, eval_time or wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12, help="LHS initial points before GP+EIps")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    history = load_history()
    print(f"[tune] resuming with {len(history)} prior trials")

    n_init = max(args.seed - len(history), 0)
    opt_kwargs = dict(
        base_estimator="GP", acq_func="EIps",
        random_state=args.random_state,
    )
    if n_init > 0:
        opt_kwargs["initial_point_generator"] = "lhs"
        opt_kwargs["n_initial_points"] = n_init
    else:
        # History already covers the seed phase; skip LHS entirely.
        opt_kwargs["n_initial_points"] = 1
    opt = Optimizer(SPACE, **opt_kwargs)
    if history:
        xs = [h[0] for h in history]
        ys = [[h[1], h[2]] for h in history]
        # Single batch tell → one GP fit instead of N.
        opt.tell(xs, ys)

    best_so_far = max((h[3] for h in history), default=-1.0)
    print(f"[tune] best Sharpe so far: {best_so_far:+.4f}")

    done = len(history)
    target = args.trials
    while done < target:
        trial_idx = done + 1
        x = opt.ask()
        params = _cast_params(x)
        print(f"\n[tune] trial {trial_idx}/{target}: {params}")
        sys.stdout.flush()
        sharpe, cost = run_trial(trial_idx, params, opt)
        if sharpe is None:
            # Failed trial — tell skopt with a bad value so it moves on
            opt.tell(x, [0.5, 120.0])  # pretend -0.5 Sharpe at 2min
            done += 1
            continue
        opt.tell(x, [-sharpe, max(cost or 60.0, 1.0)])
        done += 1
        tag = "NEW BEST" if sharpe > best_so_far else ""
        if sharpe > best_so_far:
            best_so_far = sharpe
        print(f"[tune] trial {trial_idx} done: Sharpe={sharpe:+.4f}  cost={cost:.1f}s  best={best_so_far:+.4f} {tag}")
        sys.stdout.flush()

    print(f"\n[tune] finished {done} trials. Best Sharpe = {best_so_far:+.4f}")


if __name__ == "__main__":
    main()
