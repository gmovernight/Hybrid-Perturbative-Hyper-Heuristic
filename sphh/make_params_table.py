# make_params_table.py
from pathlib import Path
import platform
from datetime import datetime

# Your experiment config
from exp_config import SEEDS, MAX_EVALS_BY_D, MAX_EVALS_DEFAULT, RESULTS_DIR

# ---- FINAL SETTINGS YOU USED (edit here only if you changed them) ----
SELECTION_MODE = "ucb"     # heuristic selection: "ucb" or "random"
ACCEPTANCE_MODE = "sa"     # acceptance: "sa" or "greedy"
UCB_C = 1.5
COOLING_FRAC = 0.2
INIT_STRATEGY = "random"
STEP_RULE = "1/5 success rule, update every 50 evals; per-dim step starts at 10% of range"
PRINT_EVERY = "max(1, max_evals // 10)"
CLAMPING = "Yes (solutions clipped to bounds)"
RUNS_PER_PROBLEM = len(SEEDS)

# Try to detect the number of low-level heuristics (optional, safe fallback)
def detect_heuristics_count():
    try:
        import numpy as np
        from single_point_sphh import SPHH
        # Dummy objective/bounds just to instantiate
        f = lambda x: float(np.sum(x*x))
        lo = -5.0; hi = 5.0; D = 10
        hh = SPHH(objective=f, bounds=(lo, hi), dim=D, max_evals=100, seed=0, verbose=False)
        return len(getattr(hh, "heuristics", []))
    except Exception:
        return "N/A"

H_COUNT = detect_heuristics_count()

OUT_MD  = RESULTS_DIR / "params.md"
OUT_TEX = RESULTS_DIR / "params.tex"

def fmt_budgets():
    if MAX_EVALS_BY_D:
        pairs = [f"D={d}: {MAX_EVALS_BY_D[d]}" for d in sorted(MAX_EVALS_BY_D)]
        return ", ".join(pairs)
    return f"default: {MAX_EVALS_DEFAULT}"

def build_markdown():
    lines = []
    lines.append("# Experiment Parameters\n")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("\n| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Seeds | {SEEDS} |")
    lines.append(f"| Runs per problem | {RUNS_PER_PROBLEM} |")
    lines.append(f"| Budget (function evaluations) | {fmt_budgets()} (default {MAX_EVALS_DEFAULT}) |")
    lines.append(f"| Selection mode | {SELECTION_MODE} |")
    lines.append(f"| Acceptance mode | {ACCEPTANCE_MODE} |")
    lines.append(f"| UCB constant (c) | {UCB_C} |")
    lines.append(f"| Cooling fraction | {COOLING_FRAC} |")
    lines.append(f"| Init strategy | {INIT_STRATEGY} |")
    lines.append(f"| Step-size rule | {STEP_RULE} |")
    lines.append(f"| Print frequency | {PRINT_EVERY} |")
    lines.append(f"| Clamping to bounds | {CLAMPING} |")
    lines.append(f"| # Low-level heuristics (H) | {H_COUNT} |")
    lines.append(f"| Python | {platform.python_version()} |")
    lines.append(f"| Platform | {platform.platform()} |")
    return "\n".join(lines) + "\n"

def build_latex():
    rows = [
        ("Seeds", f"{SEEDS}"),
        ("Runs per problem", f"{RUNS_PER_PROBLEM}"),
        ("Budget (function evaluations)", f"{fmt_budgets()} (default {MAX_EVALS_DEFAULT})"),
        ("Selection mode", SELECTION_MODE),
        ("Acceptance mode", ACCEPTANCE_MODE),
        ("UCB constant (c)", str(UCB_C)),
        ("Cooling fraction", str(COOLING_FRAC)),
        ("Init strategy", INIT_STRATEGY),
        ("Step-size rule", STEP_RULE),
        ("Print frequency", PRINT_EVERY),
        ("Clamping to bounds", CLAMPING),
        ("\\# Low-level heuristics (H)", str(H_COUNT)),
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
    ]
    lines = []
    lines.append("% Parameters table")
    lines.append("\\section*{Experiment Parameters}")
    lines.append("\\begin{tabular}{l l}")
    lines.append("\\hline")
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("")
    return "\n".join(lines)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(), encoding="utf-8")
    OUT_TEX.write_text(build_latex(), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")
    print(f"[OK] wrote {OUT_TEX}")

if __name__ == "__main__":
    main()
