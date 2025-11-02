# make_tables.py  (rows = D; columns = f_best, mean_f_best, f_best_std_dev, mean run time (s))
from pathlib import Path
import csv
import re
from collections import defaultdict

IN_FILE = Path("results") / "summary_all.csv"
OUT_MD  = Path("results") / "tables.md"
OUT_TEX = Path("results") / "tables.tex"

def base_func(key: str) -> str:
    return key.split("_")[0]  # "f3_D10" -> "f3"; "f1" -> "f1"

def parse_D(row) -> int:
    try:
        return int(row["D"])
    except:
        m = re.search(r"_D(\d+)$", row["function_key"])
        return int(m.group(1)) if m else -1

def fmt(x, nd=6):
    try:
        return f"{float(x):.{nd}g}"
    except:
        return str(x)

def load_groups():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run aggregate_suite.py first.")
    groups = defaultdict(list)
    with IN_FILE.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            groups[base_func(row["function_key"])].append(row)
    # sort each group by D (numeric; -1 or missing goes first)
    for k in groups:
        groups[k].sort(key=lambda r: parse_D(r))
    return groups

def build_markdown(groups):
    lines = []
    lines.append("# Benchmark Summary Tables\n")
    for bfunc in sorted(groups.keys(), key=lambda s: int(s[1:])):  # order by function number
        rows = groups[bfunc]
        lines.append(f"\n## {bfunc}\n")
        lines.append("| D | f_best | mean_f_best | f_best_std_dev | mean run time (s) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in rows:
            d   = parse_D(r)
            best = fmt(r["best_f_best"])
            meanb = fmt(r["mean_f_best"])
            stdb  = fmt(r["std_f_best"])
            mrt   = fmt(r["mean_runtime_s"])
            lines.append(f"| {d if d != -1 else '—'} | {best} | {meanb} | {stdb} | {mrt} |")
    return "\n".join(lines) + "\n"

def build_latex(groups):
    lines = []
    lines.append("% LaTeX tables generated from summary_all.csv")
    lines.append("\\section*{Benchmark Summary Tables}")
    for bfunc in sorted(groups.keys(), key=lambda s: int(s[1:])):  # order by function number
        rows = groups[bfunc]
        lines.append(f"\\subsection*{{{bfunc}}}")
        lines.append("\\begin{tabular}{r r r r r}")
        lines.append("\\hline")
        lines.append("D & f\\_best & mean\\_f\\_best & f\\_best\\_std\\_dev & mean run time (s) \\\\")
        lines.append("\\hline")
        for r in rows:
            d   = parse_D(r)
            best = fmt(r["best_f_best"])
            meanb = fmt(r["mean_f_best"])
            stdb  = fmt(r["std_f_best"])
            mrt   = fmt(r["mean_runtime_s"])
            d_str = f"{d}" if d != -1 else "\\text{--}"
            lines.append(f"{d_str} & {best} & {meanb} & {stdb} & {mrt} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}\n")
    return "\n".join(lines)

def main():
    groups = load_groups()
    OUT_MD.write_text(build_markdown(groups), encoding="utf-8")
    OUT_TEX.write_text(build_latex(groups), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")
    print(f"[OK] wrote {OUT_TEX}")

if __name__ == "__main__":
    main()
