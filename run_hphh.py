from __future__ import annotations

import argparse
import time

import numpy as np

from benchmark_functions import OBJECTIVES
from hphh.hphh import HPHH


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Perturbative Hyper-Heuristic (HPHH) on a benchmark function.")
    parser.add_argument("--objective", type=str, default="f13_D10", help="Objective key from benchmark_functions.OBJECTIVES")
    parser.add_argument("--max-evals", type=int, default=100_000, help="Total evaluation budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --runs=1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs; seeds will be 0..runs-1")

    # Stage fractions
    parser.add_argument("--sphh-frac", type=float, default=0.1, help="Fraction of budget for SPHH pre-optimization")
    parser.add_argument("--gp-frac", type=float, default=0.1, help="Fraction of budget for GPHH evolution")

    # SPHH controls (optional)
    parser.add_argument("--sphh-ucb-c", type=float, default=1.5)
    parser.add_argument("--sphh-cooling-frac", type=float, default=0.2)
    parser.add_argument("--sphh-selection", type=str, default="ucb", choices=["ucb", "random"])
    parser.add_argument("--sphh-acceptance", type=str, default="sa", choices=["sa", "greedy"])

    # GPHH controls (optional)
    parser.add_argument("--gp-pop", type=int, default=60)
    parser.add_argument("--gp-gens", type=int, default=20)
    parser.add_argument("--eval-budget-per-prog", type=int, default=1000)
    parser.add_argument("--tree-depth-max", type=int, default=5)
    parser.add_argument("--tournament-k", type=int, default=4)
    parser.add_argument("--p-cx", type=float, default=0.8)
    parser.add_argument("--p-mut", type=float, default=0.2)

    args = parser.parse_args()

    if args.objective not in OBJECTIVES:
        keys = ", ".join(sorted(OBJECTIVES.keys()))
        raise SystemExit(f"Unknown objective '{args.objective}'. Available: {keys}")

    f, lo, hi, D = OBJECTIVES[args.objective]

    runs = int(max(1, args.runs))
    f_bests = []
    runtimes = []

    for i in range(runs):
        seed = i if runs > 1 else args.seed
        print(f"\n=== Run {i+1}/{runs} | seed={seed} ===")
        solver = HPHH(
            objective=f,
            bounds=(lo, hi),
            dim=D,
            max_evals=args.max_evals,
            seed=seed,
            run_index=i,
            run_total=runs,
            job_label=args.objective,
            sphh_frac=args.sphh_frac,
            gp_frac=args.gp_frac,
            sphh_ucb_c=args.sphh_ucb_c,
            sphh_cooling_frac=args.sphh_cooling_frac,
            sphh_selection=args.sphh_selection,
            sphh_acceptance=args.sphh_acceptance,
            gp_pop=args.gp_pop,
            gp_gens=args.gp_gens,
            eval_budget_per_prog=args.eval_budget_per_prog,
            tree_depth_max=args.tree_depth_max,
            tournament_k=args.tournament_k,
            p_cx=args.p_cx,
            p_mut=args.p_mut,
            verbose=True,
            print_every=1,
        )

        res = solver.run()

        print("HPHH finished")
        print(f"Objective: {args.objective}")
        print(f"Best f: {res.f_best:.6g}")
        print(f"Total evaluations: {res.evaluations}")
        print(f"Stage evals – SPHH: {res.sphh_evals}, GP evolve: {res.gp_evals}, Program apply: {res.program_apply_evals}")
        print(f"Runtime (s): {res.runtime_sec:.3f}")

        f_bests.append(float(res.f_best))
        runtimes.append(float(res.runtime_sec))

    # Summary across runs
    from statistics import mean, stdev

    best = min(f_bests) if f_bests else float('nan')
    avg = mean(f_bests) if f_bests else float('nan')
    std = stdev(f_bests) if len(f_bests) > 1 else 0.0
    mean_rt = mean(runtimes) if runtimes else float('nan')

    print("\nSummary:")
    print("Function Dim Runs Best Average Std MeanRuntime(s)")
    print(f"{args.objective} {D} {runs} {best:.6g} {avg:.6g} {std:.6g} {mean_rt:.6g}")


if __name__ == "__main__":
    main()
