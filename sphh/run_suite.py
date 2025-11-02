# run_suite.py
from pathlib import Path
from datetime import datetime
import csv
import re

from single_point_sphh import SPHH, OBJECTIVES
from exp_config import SEEDS, budget_for_dim, RESULTS_DIR

OUT_FILE = RESULTS_DIR / "seeds_all.csv"

def func_num(key: str) -> int:
    # Extract numeric part after leading 'f' (e.g., 'f13_D30' -> 13)
    m = re.match(r"f(\d+)", key)
    return int(m.group(1)) if m else 9999

def dim_of_key(key: str) -> int:
    # Extract D if present (e.g., 'f13_D30' -> 30), else use the registry's dim
    m = re.search(r"_D(\d+)$", key)
    if m:
        return int(m.group(1))
    # Fallback to registry
    _, _, _, dim = OBJECTIVES[key]
    return dim

def sorted_keys():
    # Include every key in OBJECTIVES; sort by function number then dimension
    keys = list(OBJECTIVES.keys())
    # Ensure stable ordering: f1, f2, f3_D10, f3_D30, f3_D50, ...
    keys.sort(key=lambda k: (func_num(k), dim_of_key(k)))
    return keys

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header = ["function_key","D","seed","max_evals","f_best","evaluations","runtime_s","timestamp"]
    new_file = not OUT_FILE.exists()
    with OUT_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)

        for key in sorted_keys():
            func, lo, hi, dim = OBJECTIVES[key]
            max_evals = budget_for_dim(dim)

            for seed in SEEDS:
                hh = SPHH(
                    objective=func, bounds=(lo, hi), dim=dim,
                    max_evals=max_evals, seed=seed,
                    verbose=False, print_every=max(1, max_evals // 10),
                )
                res = hh.run()
                w.writerow([
                    key, dim, seed, max_evals,
                    f"{res.f_best:.10g}", res.evaluations, f"{res.runtime_sec:.6f}",
                    datetime.now().isoformat(timespec="seconds"),
                ])
                print(f"[OK] {key:8s} D={dim:>2} seed={seed} best={res.f_best:.6g}")

    print(f"[DONE] Wrote all runs to {OUT_FILE}")

if __name__ == "__main__":
    main()
