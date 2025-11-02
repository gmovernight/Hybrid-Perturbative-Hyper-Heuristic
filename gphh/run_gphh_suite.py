#!/usr/bin/env python3
"""
run_gphh_suite.py — batch runner for GPHH (disposable-only)
- AUTO runs f1..f24 including base keys (e.g., f1, f2) and any _D10/_D30/_D50 variants that exist.
- Groups logs by function with a simple ETA.
Writes a CSV of results, one row per (objective, seed) run.
"""
import argparse, os, sys, time, csv, re
from datetime import datetime

# Robust imports
try:
    from benchmark_functions import OBJECTIVES
    from gphh import GPHH
except Exception:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from benchmark_functions import OBJECTIVES
    from gphh import GPHH

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
    return 0  # base keys like f1,f2 treated as dim 0 for sorting

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
    ap.add_argument("--seeds", type=str, default="1..10", help="Seed list/range, e.g., 1..10 or 1,2,3")
    ap.add_argument("--out", type=str, default="results/gphh_results.csv", help="Output CSV path (appends)")
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--per-prog", type=int, default=3000)
    ap.add_argument("--evals", type=int, default=200000)
    ap.add_argument("--depth", type=int, default=5, help="Max tree depth")
    ap.add_argument("--tk", type=int, default=4, help="Tournament k")
    ap.add_argument("--pcx", type=float, default=0.8, help="Crossover probability")
    ap.add_argument("--pmut", type=float, default=0.2, help="Mutation probability")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
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

    header = ["algo","timestamp","obj_key","dim","seed","gp_pop","gp_gens","per_prog","evals",
              "tree_depth_max","tournament_k","p_cx","p_mut","best_f","evals_used","runtime_sec","program_str","status","error"]
    ensure_header(args.out, header)

    # Sort for nice grouping
    obj_keys = sorted(obj_keys, key=_order_key)

    total = len(obj_keys) * len(seeds)
    idx = 0
    t_start = time.time()
    times = []

    # Print a compact plan
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
        for seed in seeds:
            idx += 1
            stamp = datetime.now().isoformat(timespec="seconds")

            # ETA
            avg = (sum(times)/len(times)) if times else None
            remaining = total - (idx - 1)
            eta_txt = f", ETA~{fmt_eta(avg*remaining)}" if avg is not None else ""
            print(f"[{idx}/{total}] {obj} (D={D}) seed={seed}  pop={args.pop} gens={args.gens} per={args.per_prog} evals={args.evals}{eta_txt}")

            status = "ok"; err = ""
            best_f = float("nan"); evals_used = 0; runtime = 0.0; program_str = ""

            t0 = time.time()
            try:
                solver = GPHH(
                    f, (lo, hi), D,
                    seed=seed,
                    gp_pop=args.pop,
                    gp_gens=args.gens,
                    eval_budget_per_prog=args.per_prog,
                    max_evals=args.evals,
                    tree_depth_max=args.depth,
                    tournament_k=args.tk,
                    p_cx=args.pcx,
                    p_mut=args.pmut,
                    verbose=args.verbose,
                    print_every=args.print_every
                )
                res = solver.run()
                best_f = float(res.f_best)
                evals_used = int(res.evaluations)
                runtime = float(res.runtime_sec)
                program_str = getattr(solver, "_best_prog_desc", "")
            except Exception as e:
                status = "error"
                err = f"{type(e).__name__}: {e}"
                print(f"    !! ERROR: {err}")
            finally:
                dt = time.time() - t0
                times.append(dt)
                with open(args.out, "a", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow([
                        "GPHH", stamp, obj, D, seed, args.pop, args.gens, args.per_prog, args.evals,
                        args.depth, args.tk, args.pcx, args.pmut, best_f, evals_used, runtime, program_str, status, err
                    ])
            print(f"    -> best_f={best_f:.6g}  runtime={runtime:.2f}s  ({dt:.1f}s wall)  status={status}")

    total_dt = time.time() - t_start
    print(f"\n[done] total time: {fmt_eta(total_dt)}  results -> {args.out}")

if __name__ == "__main__":
    main()
