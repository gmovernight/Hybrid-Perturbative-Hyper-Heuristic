# Hybrid Perturbative Hyper‑Heuristic (HPHH)

> A fast, reproducible hybrid hyper‑heuristic that composes **selection‑perturbative** and **generation‑perturbative** moves to solve black‑box optimization test functions. Includes clean CLI, logging, and evaluation utilities (PAR‑2, cactus & bar plots).

![status-badge](https://img.shields.io/badge/python-3.9%2B-blue)
![license-badge](https://img.shields.io/badge/license-MIT-green)
![platforms-badge](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)

---

## Table of Contents
- [Overview](#overview)
- [Why a *Hybrid* perturbative heuristic?](#why-a-hybrid-perturbative-heuristic)
- [Features](#features)
- [Quickstart](#quickstart)
- [Command-Line Usage](#command-line-usage)
- [Reproducibility](#reproducibility)
- [Outputs & Folder Structure](#outputs--folder-structure)
- [Evaluation (PAR‑2, Cactus, Bars)](#evaluation-par2-cactus-bars)
- [Architecture in a Nutshell](#architecture-in-a-nutshell)
- [Project Structure](#project-structure)
- [Results Snapshot](#results-snapshot)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Cite This Work](#cite-this-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository contains a **Hybrid Perturbative Hyper‑Heuristic (HPHH)** for continuous black‑box benchmarks used in **COS790** Assignment 3. The hybrid approach composes two families of moves:

1. **Selection‑Perturbative (SPHH)** — choose among a set of pre‑defined local operators (e.g., Gaussian nudge, coordinate step, restart) using adaptive selection.
2. **Generation‑Perturbative (GPHH)** — *generate* candidate solutions (e.g., recombination / sampling‑based proposals) and accept based on improvement criteria.

**HPHH** combines both: at each step it decides to **select** a local operator or **generate** a candidate, balancing exploitation and exploration.

Companion baselines (from prior assignments) are included for comparison:
- **SPHH** (Selection only)
- **GPHH** (Generation only)

---

## Why a *Hybrid* perturbative heuristic?

Pure selection strategies can **over‑exploit** a local region; pure generation strategies may **wander** without enough focus. The hybrid design aims to:
- exploit local curvature **(selection)** when improvement is steady,
- jump/reshuffle **(generation)** when stagnation is detected,
- adapt the mix over time with simple statistics (improvement rates, stagnation counters).

---

## Features

- ⚙️ **Hybrid engine**: alternates or composes SPHH and GPHH moves with adaptive gating.
- 🧪 **Benchmark suite hooks**: run on function keys like `f1, f2, …` and dimensions `D10, D30, D50`.
- 🔁 **Multi‑run automation**: `--runs N` executes seeds `[0 … N-1]` automatically.
- ⏱️ **Budgets**: function‑evaluation budgets or iteration budgets via `--budget`.
- 🧾 **Structured logs**: per‑iteration bests, per‑run JSONL, and per‑family CSV summaries.
- 📈 **Evaluation utilities**: compute **PAR‑2**, render **bar** and **cactus** plots.
- 🧬 **Reproducible**: explicit seeding of Python/NumPy RNG; seeds are recorded in metadata.
- 🖥️ **Cross‑platform**: Windows / Linux / macOS.

---

## Quickstart

```bash
# 1) Clone
git clone https://github.com/gmovernight/Hybrid-Perturbative-Hyper-Heuristic.git
cd Hybrid-Perturbative-Hyper-Heuristic

# 2) (Optional) Create a virtual environment
python -m venv .venv && . .venv/bin/activate   # on Linux/macOS
# .\.venv\Scripts\activate                     # on Windows PowerShell

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run a tiny demo (hybrid, 3 runs, small budget)
python run_hphh.py --mode hybrid --keys f1 --dims 10 --runs 3 --budget 3000 --outdir results_hybrid
```

> **Tip:** The console prints the **best‑so‑far value every iteration** when `--log-every 1` is used. See [Command‑Line Usage](#command-line-usage).

---

## Command‑Line Usage

```
python run_hphh.py [--mode MODE] [--keys K [K ...]] [--dims D [D ...]]
                   [--runs N] [--budget B] [--outdir PATH]
                   [--seed S] [--log-every K] [--rng {py, numpy}]
```

**Common flags**

- `--mode {hybrid,sphh,gphh}`: choose algorithm (default: `hybrid`).
- `--keys`: which benchmark functions to run, e.g. `--keys f1 f2 f3`.
- `--dims`: problem dimensions, e.g. `--dims 10 30 50`.
- `--runs`: number of repeated runs (each run gets seed `0..runs-1`).
- `--budget`: iterations / evaluations per run (e.g., `3000`).  
- `--outdir`: where to store results (CSV/JSONL/plots).
- `--seed`: set an explicit base seed for a **single** run (overrides run‑index seeding).
- `--log-every`: print progress every *K* iterations (`1` prints **every** iteration).
- `--rng`: RNG backend, one of `{py, numpy}`; both are seeded for safety.

**Examples**

Run hybrid on f1, f3, f5 at D=10/30 with verbose iteration logging:
```bash
python run_hphh.py --mode hybrid --keys f1 f3 f5 --dims 10 30   --runs 10 --budget 3000 --log-every 1 --outdir results_hybrid
```

Compare SPHH vs GPHH vs HPHH on f1..f6 at D=10/30/50:
```bash
python run_hphh.py --mode sphh   --keys f1 f2 f3 f4 f5 f6 --dims 10 30 50 --runs 10 --budget 3000 --outdir results_sphh
python run_hphh.py --mode gphh   --keys f1 f2 f3 f4 f5 f6 --dims 10 30 50 --runs 10 --budget 3000 --outdir results_gphh
python run_hphh.py --mode hybrid --keys f1 f2 f3 f4 f5 f6 --dims 10 30 50 --runs 10 --budget 3000 --outdir results_hphh
```

---

## Reproducibility

- **Seeds per run**: if you pass `--runs N`, run `i` uses **seed `i`** (i ∈ `[0, N-1]`).  
- **Single‑run override**: pass `--seed S` to force a specific seed.
- **RNG**: both `random` (Python) and `numpy.random` are seeded; `--rng` lets you pick the main backend.  
- **Recorded**: each run writes a `metadata.json` containing `seed`, `rng`, `mode`, `keys`, `dims`, `budget`, and timestamps so experiments are fully traceable.

---

## Outputs & Folder Structure

```
results_*/
  └── <experiment_tag>/
      ├── logs/
      │   └── run_<key>_D<dim>_seed<k>.jsonl      # one line per iteration (time, best_so_far, etc.)
      ├── tables/
      │   ├── per_run.csv                         # per-run aggregates
      │   └── summary.csv                         # per-family aggregates (median, mean, PAR-2)
      ├── plots/
      │   ├── bars_par2_<key>_D<dim>.png
      │   ├── bars_median_<key>_D<dim>.png
      │   └── cactus_<key>_D<dim>.png
      └── metadata.json
```

- **JSONL logs**: include per‑iteration `best_value`, iteration, and time (s).
- **CSV**: `per_run.csv` contains run‑level metrics; `summary.csv` aggregates by family/dimension.
- **Plots**: generated by utilities in `tools/` (see below).

---

## Evaluation (PAR‑2, Cactus, Bars)

We report **Solved%**, **median time (s)**, and **PAR‑2 (s)**. In cases where all methods solve all instances, **PAR‑2 equals mean time** and cactus curves saturate at the total set size.

Generate plots from a folder of results:
```bash
# Bar plots for PAR-2 & median
python tools/plot_bars.py --results results_hphh/<tag> --metric par2_s
python tools/plot_bars.py --results results_hphh/<tag> --metric median_s

# Cactus plots
python tools/plot_cactus.py --results results_hphh/<tag>
```

> **Note:** If your environment differs, adapt the script paths. The CSV layout is documented in the scripts’ headers.

---

## Architecture in a Nutshell

**High‑level loop (pseudocode):**
```text
state ← x0
best  ← f(x0)

for t in 1..budget:
    if stagnating(best_history):
        # exploration jump (GPHH-style)
        x' ← generate_candidate(state, sampler=mixture)
    else:
        # local improvement (SPHH-style)
        op ← select_operator(stats)        # e.g., Gaussian, coordinate, n-step
        x' ← op.apply(state)

    if f(x') ≤ f(state):
        state ← x'
        best  ← min(best, f(x'))
        update_success_stats()
    else:
        update_fail_stats()

    if t % log_every == 0:
        print("[t=%d] best=%.6g" % (t, best))
```

**Key ideas**
- **Adaptive gating** toggles between selection and generation using simple stagnation tests.
- **Operator credit assignment** updates operator scores from recent improvements.
- **Restarts** (optional) reset to best or sampled points when stuck.

---

## Project Structure

```
.
├── run_hphh.py                # CLI entrypoint (HPHH/SPHH/GPHH)
├── hphh/                      # algorithm modules
│   ├── hybrid.py
│   ├── selection.py
│   ├── generation.py
│   └── utils.py
├── benchmarks/                # f1..fN functions and dimension configs
├── tools/                     # evaluation & plotting utilities
│   ├── plot_bars.py
│   └── plot_cactus.py
├── requirements.txt
└── README.md
```

> Your tree may differ slightly; the above serves as an orienting guide.

---

## Results Snapshot

Upload your generated figures and reference them here:

- **Bars (PAR‑2)**: `plots/bars_par2_f1_D10.png`
- **Bars (median)**: `plots/bars_median_f1_D10.png`
- **Cactus**: `plots/cactus_f1_D10.png`

A concise results table can be auto‑generated from `tables/summary.csv`. If you want the README to include a *live* table, consider committing a small `scripts/make_readme_table.py` that reads `summary.csv` and writes a Markdown table.

> *All functions were run over **10** runs unless otherwise stated.*

---

## Troubleshooting

- **“src refspec main does not match any”** during `git push`: make an initial commit and ensure your branch is named `main`  
  ```bash
  git add -A && git commit -m "Initial commit"
  git branch -M main
  git push -u origin main
  ```

- **Pip permissions on WSL/Debian** (`dpkg lock` error): use `sudo` for system packages, or prefer a virtualenv and `pip` from the venv:
  ```bash
  python -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt
  ```

- **No plots saved**: check you have write access to `--outdir` and that `tools/` is on `PYTHONPATH` or called with the right relative path.

---

## Roadmap

- [ ] Add operator‑ablation study.
- [ ] Add automatic per‑family operator weighting.
- [ ] Add YAML experiment configs and sweep runner.
- [ ] Export unified `all_methods_summary.csv` (SPHH/GPHH/HPHH).
- [ ] Optional Torch‑based samplers for generation moves.

---

## Cite This Work

If you use this repository or build upon it, please cite:

```bibtex
@misc{hphh2025,
  title        = {Hybrid Perturbative Hyper-Heuristic (HPHH)},
  author       = {Carlinsky, Ruan},
  howpublished = {\url{https://github.com/gmovernight/Hybrid-Perturbative-Hyper-Heuristic}},
  year         = {2025},
  note         = {COS790 Assignment 3}
}
```

You may also cite the baseline repos you adapted (SPHH, GPHH) as needed.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgments

- University of Pretoria — COS790 course staff and peers.
- Prior assignments and baseline implementations (SPHH, GPHH) that informed the hybrid design.
- Open‑source community for plotting and evaluation utilities.
