#!/usr/bin/env python3
"""
run_hphh_suite.py — batch runner for HPHH

- AUTO runs f1..f24 including base keys (e.g., f1, f2) and any _D10/_D30/_D50 variants that exist.
- Writes one CSV (append) with one row per (objective, seed) run.

Usage examples
--------------
python3 hphh/run_hphh_suite.py --auto --seeds 0..9 \
  --evals 10000 --sphh-frac 0.1 --gp-frac 0.1 \
  --pop 8 --per-prog 40 --gens 3 --out results/hphh_results.csv

Then aggregate:
python3 hphh/aggregate_hphh_results.py --input results/hphh_results.csv --output results/hphh_summary.csv
"""
import argparse, os, sys, time, csv, re
from datetime import datetime

# Ensure project root is on sys.path so we can import benchmark_functions and the hphh module file
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark_functions import OBJECTIVES
# Import HPHH class from the module file hphh/hphh.py; when running this script, sys.path includes hphh dir
from hphh import HPHH


def parse_seeds(s: str):
    s = s.strip()
    m = re.match(r"^\s*(\d+)\s*(?:\.\.|-)\s*(\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _base_of(key: str) -> str:
    return key.split("_")[0] if "_D" in key else key


def _dim_of(key: str) -> int:
    if "_D" in key:
        try:
            return int(key.split("_D", 1)[1])
        except Exception:
            return -1
    # f1,f2 base keys treated as dim from registry
    try:
        _, _, _, D = OBJECTIVES[key]
        return int(D)
    except Exception:
        return -1


def _order_key(key: str):
    try:
        b = _base_of(key)
        i = int(b[1:]) if b.startswith("f") else 999
    except Exception:
        i = 999
    return (i, _dim_of(key))


def expand_objectives(objs_str: str):
    """
    Accepts:
      - 'AUTO' (f1..f24, including base keys like f1,f2 and any _D10/_D30/_D50 present)
      - 'ALL', 'ALL_D10', 'ALL_D30', 'ALL_D50'
      - base names like 'f24' (expands to all available dimensions for that base, plus base if present)
      - patterns like 'f24_D*'
      - explicit keys 'f3_D10,f7_D30', etc.
    """
    avail = list(OBJECTIVES.keys())
    have = set(avail)
    s = objs_str.strip()

    if s.upper() == "AUTO":
        out = []
        for i in range(1, 25):
            base = f"f{i}"
            if base in have:
                out.append(base)
            for D in (10, 30, 50):
                k = f"{base}_D{D}"
                if k in have:
                    out.append(k)
        return sorted(out, key=_order_key)

    if s.upper() == "ALL":
        return sorted(avail, key=_order_key)

    m = re.match(r"^ALL_D(10|30|50)$", s, flags=re.IGNORECASE)
    if m:
        D = m.group(1)
        return sorted([k for k in avail if k.endswith(f"_D{D}")], key=_order_key)

    # Comma-separated mix
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    selected = set()

    def add_base(name: str):
        if name in have:
            selected.add(name)
        for D in (10, 30, 50):
            k = f"{name}_D{D}"
            if k in have:
                selected.add(k)

    for t in tokens:
        if t in have:
            selected.add(t)
            continue
        if re.match(r"^f\d+$", t):  # just the base name
            add_base(t)
            continue
        if t.endswith("_D*"):
            add_base(t[:-3])
            continue
        raise KeyError(f"Objective key or pattern not found: {t}")

    return sorted(selected, key=_order_key)


def ensure_header(path, header):
    exists = os.path.exists(path)
    if not exists or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)


def fmt_eta(seconds: float) -> str:
    if seconds < 0: seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", type=str, required=False,
                    help="AUTO | ALL | ALL_D10/30/50 | fK | fK_D* | explicit keys (comma-separated)")
    ap.add_argument("--auto", action="store_true", help="Run all f1..f24 across D=10,30,50 and base keys if present")
    ap.add_argument("--seeds", type=str, default="0..9", help="Seed list/range, e.g., 0..9 or 0,1,2")
    ap.add_argument("--out", type=str, default="results/hphh_results.csv", help="Output CSV path (appends)")

    # HPHH budgets
    ap.add_argument("--evals", type=int, default=10000, help="Total evaluation budget per run")
    ap.add_argument("--sphh-frac", type=float, default=0.1)
    ap.add_argument("--gp-frac", type=float, default=0.1)

    # GP params
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--per-prog", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=5, help="Max tree depth")
    ap.add_argument("--tk", type=int, default=4, help="Tournament k")
    ap.add_argument("--pcx", type=float, default=0.8, help="Crossover probability")
    ap.add_argument("--pmut", type=float, default=0.2, help="Mutation probability")
    ap.add_argument("--print-every", type=int, default=1)
    # Default to verbose True; allow disabling with --quiet
    ap.add_argument("--verbose", dest="verbose", action="store_true", default=True, help="Enable per-iteration prints and stepwise apply (default)")
    ap.add_argument("--quiet", dest="verbose", action="store_false", help="Disable verbose output")
    args = ap.parse_args()

    if args.auto and not args.objs:
        args.objs = "AUTO"
    if not args.objs:
        raise SystemExit("Either --auto or --objs must be provided.")

    try:
        obj_keys = expand_objectives(args.objs)
    except Exception as e:
        raise SystemExit(f"Could not expand objectives: {e}")

    seeds = parse_seeds(args.seeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    header = [
        "algo","timestamp","obj_key","dim","seed",
        "sphh_frac","gp_frac","gp_pop","gp_gens","per_prog","evals",
        "best_f","evals_used","runtime_sec",
        "sphh_used","gp_used","apply_used",
        "status","error"
    ]
    ensure_header(args.out, header)
    # Also prepare per-problem directory for optional per-problem CSVs
    per_problem_dir = os.path.join(os.path.dirname(args.out) or ".", "hphh_by_problem")
    os.makedirs(per_problem_dir, exist_ok=True)

    # Sort for nice grouping
    obj_keys = sorted(obj_keys, key=_order_key)

    total = len(obj_keys) * len(seeds)
    idx = 0
    t_start = time.time()
    times = []

    bases = {}
    for k in obj_keys:
        bases.setdefault(_base_of(k), []).append(k)
    print("[plan] functions:", ", ".join(f"{b}×{len(v)}" for b,v in bases.items()))
    print(f"[plan] total runs: {total}  (|objs|={len(obj_keys)} × |seeds|={len(seeds)})")
    print()

    curr_base = None
    for obj in obj_keys:
        base = _base_of(obj)
        if base != curr_base:
            curr_base = base
            dims_here = [k for k in obj_keys if _base_of(k) == base]
            print(f"=== {base} : {len(dims_here)} objective key(s) ({', '.join(dims_here)}) ===")

        f, lo, hi, D = OBJECTIVES[obj]
        for s_idx, seed in enumerate(seeds):
            idx += 1
            stamp = datetime.now().isoformat(timespec="seconds")

            avg = (sum(times)/len(times)) if times else None
            remaining = total - (idx - 1)
            eta_txt = f", ETA~{fmt_eta(avg*remaining)}" if avg is not None else ""
            print(f"[{idx}/{total}] {obj} (D={D}) seed={seed}  pop={args.pop} gens={args.gens} per={args.per_prog} evals={args.evals}{eta_txt}")

            status = "ok"; err = ""
            best_f = float("nan"); evals_used = 0; runtime = 0.0
            sphh_used = 0; gp_used = 0; apply_used = 0

            t0 = time.time()
            try:
                solver = HPHH(
                    objective=f, bounds=(lo, hi), dim=D,
                    seed=seed, max_evals=args.evals,
                    run_index=s_idx, run_total=len(seeds), job_label=obj,
                    sphh_frac=args.sphh_frac, gp_frac=args.gp_frac,
                    gp_pop=args.pop, gp_gens=args.gens, eval_budget_per_prog=args.per_prog,
                    tree_depth_max=args.depth, tournament_k=args.tk,
                    p_cx=args.pcx, p_mut=args.pmut,
                    verbose=args.verbose, print_every=args.print_every,
                )
                res = solver.run()
                best_f = float(res.f_best)
                evals_used = int(res.evaluations)
                runtime = float(res.runtime_sec)
                sphh_used = int(getattr(res, 'sphh_evals', 0))
                gp_used = int(getattr(res, 'gp_evals', 0))
                apply_used = int(getattr(res, 'program_apply_evals', 0))
            except Exception as e:
                status = "error"
                err = f"{type(e).__name__}: {e}"
                print(f"    !! ERROR: {err}")
            finally:
                dt = time.time() - t0
                times.append(dt)
                row = [
                    "HPHH", stamp, obj, D, seed,
                    args.sphh_frac, args.gp_frac, args.pop, args.gens, args.per_prog, args.evals,
                    best_f, evals_used, runtime,
                    sphh_used, gp_used, apply_used,
                    status, err
                ]
                with open(args.out, "a", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(row)
                # Per-problem CSV: results/hphh_by_problem/<obj_key>.csv
                per_path = os.path.join(per_problem_dir, f"{obj}.csv")
                ensure_header(per_path, header)
                with open(per_path, "a", newline="") as fcsv2:
                    w2 = csv.writer(fcsv2)
                    w2.writerow(row)
            print(f"    -> best_f={best_f:.6g}  runtime={runtime:.2f}s  used={evals_used} (S={sphh_used}, G={gp_used}, A={apply_used})  status={status}")

    total_dt = time.time() - t_start
    print(f"\n[done] total time: {fmt_eta(total_dt)}  results -> {args.out}")


if __name__ == "__main__":
    main()
