# How to Run — SPHH Benchmark Suite

This guide shows how to run the full experiment pipeline and the utility scripts in this repo. It’s written for **Python 3.10+** (required because of modern type hints).

---

## TL;DR (full pipeline)

```bash
# 1) (optional) create and activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) install dependencies
pip install numpy matplotlib

# 3) run all objectives across all seeds; write per-run CSV
python run_suite.py

# 4) aggregate per-run CSV into a per-function summary
python aggregate_suite.py

# 5) build Markdown/LaTeX tables from the summary
python make_tables.py

# 6) record the experiment parameters you used (Markdown + LaTeX)
python make_params_table.py

# 7) (optional) plot convergence curves and write CSV traces
python plot_convergence.py
```

Outputs are written to the **`results/`** folder (created automatically). See [Outputs](#outputs) for details.

---

## What each script does

- **`single_point_sphh.py`** — Defines the SPHH optimizer, benchmark functions, and the registry `OBJECTIVES`. You can also run this file directly to watch a **single run** print every iteration (good for debugging).
- **`exp_config.py`** — Central place for **seeds** and **evaluation budgets** per dimension, and the `results/` directory.
- **`run_suite.py`** — Runs **every objective key** in `OBJECTIVES` for each seed from `exp_config`, writing one CSV row per run to `results/seeds_all.csv`.
- **`aggregate_suite.py`** — Reads `seeds_all.csv`, groups by function key, and writes `results/summary_all.csv` with mean/std/best and runtime stats.
- **`make_tables.py`** — Converts `summary_all.csv` into publication‑ready tables: `results/tables.md` and `results/tables.tex` (one table per base function, rows by dimension).
- **`make_params_table.py`** — Captures the **parameters you used** (selection/acceptance modes, budgets, etc.) and writes `results/params.md` and `results/params.tex`.
- **`plot_convergence.py`** — Re‑runs selected base functions across seeds, computes the **median optimality gap** curve per dimension, writes **PNG plots** and **CSV traces**.

---

## Setup

1) **Python**: 3.10+ recommended.  
2) **Packages**: `numpy`, `matplotlib`  
3) **Repo root**: Run all commands from the folder that contains these `.py` files.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install numpy matplotlib
```

---

## Run the full experiment

### 1) Suite of runs (per objective × per seed)
```bash
python run_suite.py
```
- Reads `SEEDS` and budgets from **`exp_config.py`**.
- Iterates over all keys in `OBJECTIVES`, runs SPHH, and appends rows to **`results/seeds_all.csv`**.

### 2) Aggregate into per‑function summary
```bash
python aggregate_suite.py
```
- Reads **`results/seeds_all.csv`** and writes **`results/summary_all.csv`** with mean/std/best and runtime summaries.

### 3) Build tables (Markdown + LaTeX)
```bash
python make_tables.py
```
- Produces **`results/tables.md`** and **`results/tables.tex`** grouped by base function (`f3`, `f4`, …) with rows by `D`.

### 4) Record the parameters you used
```bash
python make_params_table.py
```
- Produces **`results/params.md`** and **`results/params.tex`** listing seeds, budgets, selection/acceptance, etc.

### 5) (Optional) Convergence plots
```bash
python plot_convergence.py
```
- By default plots **`f24`**. Edit the list `BASE_FUNCTIONS` inside `plot_convergence.py` to plot others (e.g., `["f17","f21","f23","f4"]`).
- Outputs PNGs to **`results/plots/`** and median optimality‑gap traces to **`results/traces/`**.

---

## Customization

### Seeds & budgets
Edit **`exp_config.py`**:
- `SEEDS = list(range(10))` → change number or values of seeds.
- `MAX_EVALS_BY_D` → per‑dimension evaluation budgets (fallback: `MAX_EVALS_DEFAULT`).

### Which objectives run
- The **suite** (`run_suite.py`) runs **every** key in `OBJECTIVES` (e.g., `f3_D10`, `f3_D30`, …). To restrict the set, either:
  - Temporarily comment out entries in `OBJECTIVES` inside `single_point_sphh.py`, or
  - Add your own filtering inside `sorted_keys()` in `run_suite.py`.
  
### SPHH modes and verbosity
- `run_suite.py` constructs `SPHH(...)` using defaults `selection_mode="ucb"` and `acceptance_mode="sa"` and `verbose=False`.
- To change these **for the suite**, pass arguments in the `SPHH(...)` call inside `run_suite.py`.
- To watch a single verbose run, execute:
  ```bash
  python single_point_sphh.py
  ```
  and adjust `which`, `max_evals`, `verbose`, `print_every` at the bottom of that file.

### Convergence plots
- Edit `BASE_FUNCTIONS` in **`plot_convergence.py`**.
- If a base function has a non‑zero optimum `f*`, add/update its entry in `FSTAR_BY_BASE` to get a correct optimality‑gap curve.

---

## Outputs

The pipeline creates this tree (paths relative to repo root):

```
results/
  seeds_all.csv         # one row per (objective key, seed) run
  summary_all.csv       # aggregated stats per function key
  tables.md             # Markdown tables (by base function × D)
  tables.tex            # LaTeX tables
  params.md             # experiment parameters (human‑readable)
  params.tex            # LaTeX version of parameters
  plots/
    conv_gap_<base>.png # log‑scale optimality‑gap plots by D
  traces/
    trace_gap_<key>.csv # median optimality‑gap vs eval index
```

---

## Minimal troubleshooting

- **`ModuleNotFoundError: numpy/matplotlib`** → run `pip install numpy matplotlib` in your active environment.
- **`Missing results/seeds_all.csv`** when aggregating → run the suite first (`python run_suite.py`).
- **Long runtime** → reduce `SEEDS` or budgets in `exp_config.py`, or restrict the set of objective keys.

---

## Re‑running from scratch

If you want a clean slate, you can remove `results/` and re‑run the steps:

```bash
rm -rf results
python run_suite.py
python aggregate_suite.py
python make_tables.py
python make_params_table.py
python plot_convergence.py   # optional
```
