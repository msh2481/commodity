"""Analyze tune.py results: top-K trials, best-so-far, plus skopt plots.

Run with --plots to also write convergence / evaluations / objective PNGs to
_tune/plots/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TUNE_DIR = Path("/root/commodity/submissions/gbdt_per_target/_tune")
TRIALS = TUNE_DIR / "trials.jsonl"
PLOT_DIR = TUNE_DIR / "plots"

# Must mirror tune.py SPACE (same order & types)
from skopt.space import Real, Integer
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


def load():
    rows = []
    for line in TRIALS.read_text().splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        if d.get("status") != "done": continue
        rows.append(d)
    return rows


def text_summary(rows):
    rows_sorted = sorted(rows, key=lambda r: r["overall_sharpe"], reverse=True)
    print(f"N trials = {len(rows)}")
    print(f"Best Sharpe = {rows_sorted[0]['overall_sharpe']:+.4f}  (trial {rows_sorted[0]['trial']})")
    print(f"Mean Sharpe = {np.mean([r['overall_sharpe'] for r in rows]):+.4f}")
    print(f"Top Sharpe >= 0.20 count = {sum(1 for r in rows if r['overall_sharpe'] >= 0.20)}")
    print(f"Top Sharpe >= 0.23 count = {sum(1 for r in rows if r['overall_sharpe'] >= 0.23)}")

    print("\n=== Top 15 by Sharpe ===")
    print(f"{'rank':>4} {'trial':>5} {'sharpe':>7} {'ne':>4} {'md':>3} {'lr':>6} {'mcw':>4} "
          f"{'ss':>5} {'cs':>5} {'rl':>7} {'ra':>6} {'g':>6}  cost_s")
    for i, r in enumerate(rows_sorted[:15]):
        p = r["params"]
        print(f"{i+1:>4} {r['trial']:>5} {r['overall_sharpe']:+7.4f} "
              f"{p['n_estimators']:>4} {p['max_depth']:>3} {p['learning_rate']:.4f} {p['min_child_weight']:>4} "
              f"{p['subsample']:.3f} {p['colsample_bytree']:.3f} "
              f"{p['reg_lambda']:7.3f} {p['reg_alpha']:6.3f} {p['gamma']:6.4f}  {r.get('eval_time_seconds', 0):.1f}")

    best = -1.0
    print("\n=== Best-so-far milestones ===")
    for r in sorted(rows, key=lambda r: r["trial"]):
        if r["overall_sharpe"] > best:
            best = r["overall_sharpe"]
            print(f"  after trial {r['trial']:>3}: {best:+.4f}")


def build_result(rows):
    """Rebuild an OptimizeResult so skopt.plots can consume it."""
    from skopt import Optimizer
    opt = Optimizer(SPACE, base_estimator="GP", acq_func="EI", random_state=0)
    xs = []
    ys = []
    for r in sorted(rows, key=lambda r: r["trial"]):
        p = r["params"]
        try:
            x = [p[k] for k in NAMES]
        except KeyError:
            continue
        xs.append(x)
        ys.append(-r["overall_sharpe"])  # skopt minimizes
    if not xs:
        return None
    opt.tell(xs, ys)
    return opt.get_result()


def save_plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skopt.plots import plot_convergence, plot_evaluations, plot_objective

    PLOT_DIR.mkdir(exist_ok=True)
    result = build_result(rows)
    if result is None:
        print("No plottable rows."); return

    # Convergence: best-so-far over iterations
    ax = plot_convergence(result)
    fig = ax.get_figure()
    fig.suptitle("Convergence (-Sharpe, minimized)")
    fig.tight_layout()
    p = PLOT_DIR / "convergence.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"wrote {p}")

    def _fig_from_axes(axes):
        arr = np.atleast_2d(np.asarray(axes))
        return arr.flat[0].figure

    # Evaluations: pairwise scatter of each hparam sample
    axes = plot_evaluations(result, bins=20)
    fig = _fig_from_axes(axes)
    fig.suptitle("Sampled points per-dim + pairwise scatter")
    fig.tight_layout()
    p = PLOT_DIR / "evaluations.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"wrote {p}")

    # Partial-dependence / surrogate
    axes = plot_objective(result, n_samples=40, sample_source="result")
    fig = _fig_from_axes(axes)
    fig.suptitle("Partial-dependence of surrogate (higher = better Sharpe)")
    fig.tight_layout()
    p = PLOT_DIR / "objective.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    rows = load()
    if not rows:
        print("No done trials."); return
    text_summary(rows)
    if args.plots:
        print()
        save_plots(rows)


if __name__ == "__main__":
    main()
