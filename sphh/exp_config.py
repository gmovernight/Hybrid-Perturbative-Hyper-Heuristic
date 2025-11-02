# exp_config.py
from pathlib import Path

# --- Reproducibility ---
SEEDS = list(range(10))  # the 10 independent runs: 0..9

# --- Budgets ---
MAX_EVALS_DEFAULT = 5000
# Keep one budget per dimension so you can change it in ONE place later if needed.
MAX_EVALS_BY_D = {
    10: 30000,
    30: 30000,
    50: 30000,
}

def budget_for_dim(D: int) -> int:
    return MAX_EVALS_BY_D.get(D, MAX_EVALS_DEFAULT)

# --- Output location ---
RESULTS_DIR = Path("results")
