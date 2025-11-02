# HOW TO RUN — Hybrid Perturbative Hyper‑Heuristic (HPHH)

This guide shows you how to run the **Hybrid Perturbative Hyper‑Heuristic (HPHH)** experiments, reproduce the paper/suite results, and generate summaries and plots.

---

## 0) Requirements

- **Python**: 3.10+ (tested with CPython 3.12)
- **OS**: Linux/macOS/Windows
- **Python packages**: `numpy`, `pandas`, `matplotlib` (and the Python standard library)

> If you do not have a `requirements.txt`, install the minimal set:
>
> ```bash
> pip install numpy pandas matplotlib
> ```

---

## 1) Repository layout (place the three components side‑by‑side)

Make sure the **SPHH** and **GPHH** code directories are available, because HPHH imports them:

```
project/
├─ benchmark_functions.py
├─ sphh/                     # from single_point_sphh.zip
│   └─ single_point_sphh.py
├─ gphh/                     # from gphh.zip
│   └─ gphh.py
├─ hphh/                     # from hphh (2).zip
│   ├─ hphh.py
│   ├─ run_hphh_suite.py
│   ├─ aggregate_hphh_results.py
│   └─ plot_convergence.py
└─ run_hphh.py               # single‑objective runner (root)
```

> **Tip (imports):** set your Python path to the project root so `sphh` and `gphh` can be imported by HPHH.
>
> - Linux/macOS (bash/zsh): `export PYTHONPATH=$(pwd)`  
> - Windows (PowerShell): `$env:PYTHONPATH = (Get-Location)`  
> - Windows (cmd): `set PYTHONPATH=%cd%`

---

## 2) Quick start — run the full suite

Evaluate **all objectives** with 10 seeds and a **10k** evaluation budget per run. Results are appended to a CSV; then we aggregate.

```bash
# From the project root:
export PYTHONPATH=$(pwd)        # see import tip above (use your shell's equivalent)

python3 hphh/run_hphh_suite.py --auto --seeds 0..9   --evals 10000 --sphh-frac 0.1 --gp-frac 0.1   --out results/hphh_results.csv

# Summarize per objective (best/median/mean/std + runtimes)
python3 hphh/aggregate_hphh_results.py   --input results/hphh_results.csv   --output results/hphh_summary.csv
```

**Outputs** (created if missing):
- `results/hphh_results.csv` — one row per (objective, seed) run.
- `results/hphh_summary.csv` — per‑objective summary used in the paper/tables.

---

## 3) Convergence plots (optional)

This builds median optimality‑gap curves (per base function) across available dimensions and seeds.

```bash
python3 hphh/plot_convergence.py --auto --seeds 0..9   --evals 10000 --sphh-frac 0.1 --gp-frac 0.1   --pop 60 --per-prog 1000 --gens 20
```

**Outputs**:
- Figures in `results/plots_hphh/` (e.g., `conv_gap_f21.png`)
- CSV traces in `results/traces_hphh/`

---

## 4) Single objective run

Use the convenience runner to test a specific function/dimension. Example: Schwefel 2.26 in 50D.

```bash
python3 run_hphh.py   --objective f21_D50   --max-evals 10000   --seed 0   --sphh-frac 0.1 --gp-frac 0.1   --gp-pop 60 --gp-gens 20 --eval-budget-per-prog 1000
```

Key flags (common):
- `--objective`: key from `benchmark_functions.py` (e.g., `f1`, `f13_D10`, `f21_D50`)
- `--max-evals` / `--evals`: total evaluation budget
- `--sphh-frac`, `--gp-frac`: stage fractions (remainder goes to APPLY)
- `--gp-pop`, `--gp-gens`, `--per-prog`/`--eval-budget-per-prog`: GP evolution controls

---

## 5) Recommended configurations

- **Paper settings / reproducibility**: `--evals 10000 --sphh-frac 0.1 --gp-frac 0.1 --seeds 0..9`
- **If you want more APPLY (less evolution)**: increase `--sphh-frac` or reduce `--gp-frac` (e.g., `--gp-frac 0.02`) and/or reduce `--per-prog` so that GP still has >1 candidate in the population.
- **If you want real GP evolution**: keep `--evals` fixed but lower `--per-prog` to allow **pop × gens ≥ 16×5** within the GP sub‑budget (for example, `--gp-frac 0.2 --per-prog 200 --pop 16 --gens 5`).

---

## 6) Troubleshooting

- **ImportError: No module named `sphh` or `gphh`**  
  Ensure the folder structure matches the tree above and set `PYTHONPATH` to the project root.

- **No output CSV created**  
  Check that the `--out` directory exists or let the script create it; make sure you have write permissions.

- **Very slow runs**  
  The GP stage can be expensive if you use large `--pop × --per-prog × --gens`. For the paper defaults, the suite uses `--gp-frac 0.1` so the APPLY phase remains dominant.

---

## 7) File checklist

- **Inputs**: `benchmark_functions.py` (included), plus the `sphh/` and `gphh/` folders.
- **Scripts**: `hphh/run_hphh_suite.py`, `hphh/aggregate_hphh_results.py`, `hphh/plot_convergence.py`, `run_hphh.py`
- **Outputs**: `results/hphh_results.csv`, `results/hphh_summary.csv`, `results/plots_hphh/`, `results/traces_hphh/`

---

## 8) Example: full pipeline in one go

```bash
# 1) Run the suite
python3 hphh/run_hphh_suite.py --auto --seeds 0..9   --evals 10000 --sphh-frac 0.1 --gp-frac 0.1   --out results/hphh_results.csv

# 2) Aggregate
python3 hphh/aggregate_hphh_results.py   --input results/hphh_results.csv   --output results/hphh_summary.csv

# 3) Plots (optional)
python3 hphh/plot_convergence.py --auto --seeds 0..9   --evals 10000 --sphh-frac 0.1 --gp-frac 0.1   --pop 60 --per-prog 1000 --gens 20
```

> All objectives were evaluated with 10 runs (seeds `0..9`) under a 10k evaluation budget in the paper settings.
