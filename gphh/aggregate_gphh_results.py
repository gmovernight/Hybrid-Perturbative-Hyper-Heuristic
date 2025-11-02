#!/usr/bin/env python3
"""
aggregate_gphh_results.py — summarize a results CSV across seeds.
Outputs a per-objective summary CSV.
"""
import argparse, os, sys
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/gphh_results.csv", help="Path to results CSV")
    ap.add_argument("--output", type=str, default="results/gphh_summary.csv", help="Output summary CSV")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] input CSV not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    # keep ok rows for stats, but count errors too
    errors = df[df["status"] != "ok"].groupby(["obj_key"]).size().rename("error_runs")
    ok = df[df["status"] == "ok"].copy()

    # Basic stats per obj_key
    grp = ok.groupby(["obj_key", "dim"])
    summary = grp.agg(
        runs=("best_f", "size"),
        best=("best_f", "min"),
        median=("best_f", "median"),
        mean=("best_f", "mean"),
        std=("best_f", "std"),
        mean_runtime_s=("runtime_sec", "mean"),
        median_runtime_s=("runtime_sec", "median"),
    ).reset_index()

    # Attach error counts (fill missing with 0)
    summary = summary.merge(errors, how="left", left_on="obj_key", right_index=True)
    summary["error_runs"] = summary["error_runs"].fillna(0).astype(int)

    # order columns
    summary = summary[[
        "obj_key","dim","runs","error_runs","best","median","mean","std","mean_runtime_s","median_runtime_s"
    ]].sort_values(["dim","obj_key"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output}")
    # Optional: print a quick view
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary)

if __name__ == "__main__":
    main()
