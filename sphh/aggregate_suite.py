# aggregate_suite.py
from pathlib import Path
import csv
import re
from statistics import mean, stdev
from collections import defaultdict

IN_FILE  = Path("results") / "seeds_all.csv"
OUT_FILE = Path("results") / "summary_all.csv"

def func_num(key: str) -> int:
    m = re.match(r"f(\d+)", key)
    return int(m.group(1)) if m else 9999

def dim_of_key(key: str, fallback_dim: int | None = None) -> int:
    m = re.search(r"_D(\d+)$", key)
    return int(m.group(1)) if m else (fallback_dim if fallback_dim is not None else -1)

def fmt(x): 
    return f"{x:.10g}"

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run run_suite.py first.")

    groups = defaultdict(list)
    with IN_FILE.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            groups[row["function_key"]].append(row)

    if not groups:
        raise RuntimeError("No data rows found in seeds_all.csv.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "function_num","function_key","D","runs","max_evals",
        "mean_f_best","std_f_best","best_f_best",
        "mean_runtime_s","min_runtime_s","max_runtime_s"
    ]
    # Overwrite any existing summary to keep it clean
    with OUT_FILE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        # Sort by function number then dimension
        for key in sorted(groups.keys(), key=lambda k: (func_num(k), dim_of_key(k))):
            rows = groups[key]
            D = int(rows[0]["D"])
            runs = len(rows)
            max_evals = int(rows[0]["max_evals"])
            f_bests = [float(r["f_best"]) for r in rows]
            runtimes = [float(r["runtime_s"]) for r in rows]

            w.writerow({
                "function_num": func_num(key),
                "function_key": key,
                "D": D,
                "runs": runs,
                "max_evals": max_evals,
                "mean_f_best": fmt(mean(f_bests)),
                "std_f_best": fmt(stdev(f_bests)) if runs > 1 else "0",
                "best_f_best": fmt(min(f_bests)),
                "mean_runtime_s": fmt(mean(runtimes)),
                "min_runtime_s": fmt(min(runtimes)),
                "max_runtime_s": fmt(max(runtimes)),
            })

    print(f"[OK] wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
