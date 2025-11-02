#!/usr/bin/env python3
"""
aggregate_hphh_results.py — summarize HPHH results across seeds.
Reads results CSV (one row per run) and writes a per-objective summary CSV.

Columns produced:
  obj_key, dim, runs, best, median, mean, std, mean_runtime_s, median_runtime_s
"""
import argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/hphh_results.csv", help="Path to input results CSV")
    ap.add_argument("--output", type=str, default="results/hphh_summary.csv", help="Path to output summary CSV")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[error] input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    # Consider only successful runs for stats
    ok = df[df.get("status", "ok") == "ok"].copy()

    grp = ok.groupby(["obj_key", "dim"], dropna=False)
    summary = grp.agg(
        runs=("best_f", "size"),
        best=("best_f", "min"),
        median=("best_f", "median"),
        mean=("best_f", "mean"),
        std=("best_f", "std"),
        mean_runtime_s=("runtime_sec", "mean"),
        median_runtime_s=("runtime_sec", "median"),
    ).reset_index()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output}")
    print(summary)


if __name__ == "__main__":
    main()

