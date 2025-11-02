# plot_convergence.py
# Convergence plots using LOG-SCALED OPTIMALITY GAP so tiny improvements near 0 are visible.
# For each selected base function (e.g., "f3"), we:
#   - run all its D-variants (e.g., f3_D10, f3_D30, f3_D50) across all SEEDS
#   - compute the median best-so-far curve
#   - convert to |f - f*| (optimality gap), clamp to EPS, and plot with a log y-axis
# Also writes CSV traces of the median optimality gap vs evaluations.

from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

from single_point_sphh import SPHH, OBJECTIVES
from exp_config import SEEDS, budget_for_dim, RESULTS_DIR

# --- Choose the functions to plot (change this list if you want different ones) ---
BASE_FUNCTIONS = ["f24"]

# --- Known global optima f* by base function name; defaults to 0.0 if not listed ---
FSTAR_BY_BASE = {
    "f1": 0.0,
    "f2": -1.031628453,   # known nonzero optimum
    "f3": 0.0,  "f4": 0.0,  "f5": 0.0,  "f6": 0.0,
    "f7": 0.0,  "f8": 0.0,  "f9": 0.0,  "f10": 0.0,
    "f11": 0.0, "f12": 0.0, "f13": 0.0, "f14": 0.0,
    "f15": 0.0, "f16": 0.0, "f17":  -78.3323, "f18": 0.0,
    "f19": 0.0, "f20": 0.0, "f21": 0.0, "f22": 0.0,
    "f23": 0.0, "f24": 0.0,
}

EPS = 1e-12
PLOTS_DIR = RESULTS_DIR / "plots"
TRACES_DIR = RESULTS_DIR / "traces"

def base_name(key: str) -> str:
    return key.split("_")[0]

def fstar_for_key(key: str) -> float:
    return FSTAR_BY_BASE.get(base_name(key), 0.0)

def extract_D(key: str) -> int:
    m = re.search(r"_D(\d+)$", key)
    return int(m.group(1)) if m else -1

def keys_for_base(base: str):
    """Return D-variant keys for a base (e.g., f3 -> [f3_D10, f3_D30, f3_D50]),
    or [base] if no variants exist but the base key is present."""
    ks = [k for k in OBJECTIVES if k.startswith(base + "_D")]
    if ks:
        ks.sort(key=extract_D)
        return ks
    return [base] if base in OBJECTIVES else []

def run_one(key: str, seed: int) -> np.ndarray:
    """Run one seed and return best-so-far history as a 1D numpy array."""
    func, lo, hi, dim = OBJECTIVES[key]
    max_evals = budget_for_dim(dim)
    hh = SPHH(
        objective=func,
        bounds=(lo, hi),
        dim=dim,
        max_evals=max_evals,
        seed=seed,
        verbose=False,
        print_every=max(1, max_evals // 10),
    )
    res = hh.run()
    return np.asarray(res.history_best, dtype=float)

def unify_lengths(arrs):
    """Truncate all arrays to the shortest length so we can stack safely."""
    min_len = min(len(a) for a in arrs)
    return np.stack([a[:min_len] for a in arrs], axis=0)

def save_trace_csv(out_path: Path, xvals: np.ndarray, yvals: np.ndarray):
    """Save a 2-column CSV: eval_index, median_opt_gap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_index", "median_opt_gap"])
        for i, y in enumerate(yvals):
            w.writerow([int(xvals[i]), f"{y:.10g}"])

def plot_one_base(base: str):
    keys = keys_for_base(base)
    if not keys:
        print(f"[SKIP] No keys for base {base}")
        return

    med_curves = []
    labels = []
    x_axis = None

    for key in keys:
        histories = []
        for seed in SEEDS:
            hist = run_one(key, seed)
            histories.append(hist)

        H = unify_lengths(histories)             # shape: (S, T)
        median_curve = np.median(H, axis=0)      # length T

        fstar = fstar_for_key(key)
        gap = np.abs(median_curve - fstar)
        gap = np.maximum(gap, EPS)               # avoid log(0)
        med_curves.append(gap)

        d_val = extract_D(key)
        if d_val == -1:
            # Fallback to registry dim
            d_val = OBJECTIVES[key][3]
        labels.append(f"D={d_val}")

        if x_axis is None:
            x_axis = np.arange(1, len(median_curve) + 1, dtype=int)

        # Save per-key median optimality gap trace
        save_trace_csv(TRACES_DIR / f"trace_gap_{key}.csv", x_axis, gap)

    # Plot all D-variant median curves (optimality gap) on log y-axis
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for curve, lbl in zip(med_curves, labels):
        plt.semilogy(x_axis, curve, label=lbl)
    plt.xlabel("Function evaluations")
    plt.ylabel("Optimality gap |f - f*| (log scale)")
    plt.title(f"Convergence — {base}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out_png = PLOTS_DIR / f"conv_gap_{base}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Wrote {out_png}")

def main():
    for base in BASE_FUNCTIONS:
        plot_one_base(base)
    print(f"[DONE] Plots in {PLOTS_DIR} and CSV traces in {TRACES_DIR}")

if __name__ == "__main__":
    main()
