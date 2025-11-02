from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional

import numpy as np


# External components reused from SPHH and GPHH implementations
from sphh.single_point_sphh import SPHH as _SPHH
import gphh.gphh as _G


Objective = Callable[[np.ndarray], float]
Bounds = Tuple[np.ndarray, np.ndarray]


@dataclass
class Result:
    x_best: np.ndarray
    f_best: float
    evaluations: int
    runtime_sec: float
    history_best: List[float]
    
    # Optional stage breakdown
    sphh_evals: int = 0
    gp_evals: int = 0
    program_apply_evals: int = 0


class HPHH:
    """
    Hybrid Perturbative Hyper-Heuristic (HPHH)

    Design: staged hybrid that combines SPHH (Selection-based) and GPHH (Generative):
      1) SPHH pre-optimization to quickly obtain a strong incumbent solution.
      2) GPHH evolution of a macro-heuristic program within a GP budget.
      3) Apply the evolved GPHH program starting from the SPHH incumbent for the remaining budget.

    All stages operate on the same objective and bounds from benchmark_functions.py.
    """

    def __init__(
        self,
        objective: Objective,
        bounds: Bounds,
        dim: int,
        *,
        max_evals: int = 200_000,
        seed: Optional[int] = None,
        run_index: Optional[int] = None,
        job_label: Optional[str] = None,
        run_total: Optional[int] = None,
        # Stage budget fractions (must sum <= 1.0; remainder goes to Stage 3)
        sphh_frac: float = 0.1,
        gp_frac: float = 0.1,
        # SPHH params (kept close to upstream defaults)
        sphh_ucb_c: float = 1.5,
        sphh_cooling_frac: float = 0.2,
        sphh_selection: str = "ucb",
        sphh_acceptance: str = "sa",
        # GPHH params (kept close to upstream defaults)
        gp_pop: int = 60,
        gp_gens: int = 20,
        eval_budget_per_prog: int = 1000,
        tree_depth_max: int = 5,
        tournament_k: int = 4,
        p_cx: float = 0.8,
        p_mut: float = 0.2,
        verbose: bool = False,
        print_every: int = 1,
    ) -> None:
        # Problem
        self.obj: Objective = objective
        self.lo, self.hi = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.D: int = int(dim)

        # Budgets
        self.max_evals = int(max_evals)
        self.sphh_frac = float(sphh_frac)
        self.gp_frac = float(gp_frac)

        # RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.run_index = run_index
        self.job_label = job_label
        self.run_total = run_total

        # SPHH cfg
        self.sphh_ucb_c = float(sphh_ucb_c)
        self.sphh_cooling_frac = float(sphh_cooling_frac)
        self.sphh_selection = str(sphh_selection)
        self.sphh_acceptance = str(sphh_acceptance)

        # GPHH cfg
        self.gp_pop = int(gp_pop)
        self.gp_gens = int(gp_gens)
        self.eval_budget_per_prog = int(eval_budget_per_prog)
        self.tree_depth_max = int(tree_depth_max)
        self.tournament_k = int(tournament_k)
        self.p_cx = float(p_cx)
        self.p_mut = float(p_mut)

        # Reporting
        self.verbose = bool(verbose)
        self.print_every = int(print_every)

    # -------------------------- Public API --------------------------
    def run(self) -> Result:
        t0 = time.time()
        run_no = (int(self.run_index) + 1) if (self.run_index is not None) else 1
        run_tag = f"{run_no}/{self.run_total}" if (self.run_total is not None and self.run_total > 0) else f"{run_no}"
        obj_tag = f"[OBJ {self.job_label}] " if self.job_label else ""

        # Derive stage budgets
        sphh_budget = int(max(0, round(self.sphh_frac * self.max_evals)))
        gp_budget = int(max(0, round(self.gp_frac * self.max_evals)))

        if self.verbose:
            print(f"{obj_tag}[RUN {run_tag} seed={self.seed}] HPHH: total_budget={self.max_evals} | sphh_budget={sphh_budget} | gp_budget={gp_budget} | apply_budget={max(0, self.max_evals - sphh_budget - gp_budget)}")

        # Stage 1: SPHH pre-optimization
        sphh_res = self._stage_sphh(sphh_budget)

        # Stage 2: GPHH evolution (no final application here)
        best_prog, best_prog_desc, gp_used = self._stage_gphh_evolve(gp_budget)

        # Stage 3: Apply evolved program starting from SPHH incumbent
        remaining = max(0, self.max_evals - sphh_res.evaluations - gp_used)
        prog_res = self._apply_program_from_incumbent(best_prog, sphh_res.x_best, sphh_res.f_best, remaining)

        # Collate results
        total_evals = sphh_res.evaluations + gp_used + prog_res.evaluations
        history = []
        history.extend(sphh_res.history_best)
        # Align history continuity by appending program stage bests with offset
        if prog_res.history_best:
            # Ensure we don't duplicate the starting best
            if history and prog_res.history_best:
                if prog_res.history_best[0] == prog_res.history_best[0]:
                    pass
            history.extend(prog_res.history_best)

        # Choose global best between stages
        if prog_res.f_best <= sphh_res.f_best:
            x_best, f_best = prog_res.x_best, prog_res.f_best
        else:
            x_best, f_best = sphh_res.x_best, sphh_res.f_best

        return Result(
            x_best=np.asarray(x_best),
            f_best=float(f_best),
            evaluations=int(total_evals),
            runtime_sec=float(time.time() - t0),
            history_best=history,
            sphh_evals=int(sphh_res.evaluations),
            gp_evals=int(gp_used),
            program_apply_evals=int(prog_res.evaluations),
        )

    # ----------------------- Internal: Stage 1 ----------------------
    def _stage_sphh(self, budget: int):
        if budget <= 0:
            # Return a degenerate result from a single random eval to seed incumbent
            x0 = self.rng.uniform(self.lo, self.hi)
            f0 = float(self.obj(x0))
            return Result(x_best=x0, f_best=f0, evaluations=1, runtime_sec=0.0, history_best=[f0])

        run_no = (int(self.run_index) + 1) if (self.run_index is not None) else 1
        run_tag = f"{run_no}/{self.run_total}" if (self.run_total is not None and self.run_total > 0) else f"{run_no}"
        sphh = _SPHH(
            objective=self.obj,
            bounds=(self.lo, self.hi),
            dim=self.D,
            max_evals=int(budget),
            seed=int(self.rng.integers(0, 10_000_000)),
            init="random",
            cooling_frac=self.sphh_cooling_frac,
            ucb_c=self.sphh_ucb_c,
            verbose=self.verbose,
            print_every=1,
            print_prefix=f"{('[OBJ ' + self.job_label + '] ') if self.job_label else ''}[RUN {run_tag} seed={self.seed} | SPHH]",
            selection_mode=self.sphh_selection,
            acceptance_mode=self.sphh_acceptance,
        )
        res = sphh.run()
        return res

    # ----------------------- Internal: Stage 2 ----------------------
    def _stage_gphh_evolve(self, gp_budget: int):
        """Run only the GP evolution to produce a program within gp_budget; return (best_prog, desc, evals_used)."""
        if gp_budget <= 0:
            # Fallback: single primitive operator as trivial program
            prog = _G.Apply("GAUSS_FULL", {"sigma_rel": 0.1})
            return prog, _G.GPHH._program_to_str(prog), 0
        # Adjust population to respect budget for initial evaluation too
        # Ensure at least 1 program is evaluated if budget > 0
        total_budget_hint = int(max(0, gp_budget))
        safe_pop = int(max(1, min(self.gp_pop, (total_budget_hint // max(1, self.eval_budget_per_prog)) or 1)))

        solver = _G.GPHH(
            objective=self.obj,
            bounds=(self.lo, self.hi),
            dim=self.D,
            max_evals=1,  # not used here; we won't call _apply_best_program
            seed=int(self.rng.integers(0, 10_000_000)),
            gp_pop=safe_pop,
            gp_gens=self.gp_gens,
            eval_budget_per_prog=self.eval_budget_per_prog,
            tree_depth_max=self.tree_depth_max,
            tournament_k=self.tournament_k,
            p_cx=self.p_cx,
            p_mut=self.p_mut,
            verbose=False,
            print_every=max(1, self.print_every),
        )

        # Initialize and evaluate initial population; may consume most or all of gp_budget
        solver._init_population()
        solver._fitness = np.array([solver._evaluate_program(p) for p in solver._pop], dtype=float)
        best_idx = int(np.argmin(solver._fitness))
        best_prog = solver._pop[best_idx]
        best_fit = float(solver._fitness[best_idx])
        run_no = (int(self.run_index) + 1) if (self.run_index is not None) else 1
        run_tag = f"{run_no}/{self.run_total}" if (self.run_total is not None and self.run_total > 0) else f"{run_no}"
        if self.verbose:
            print(f"{('[OBJ ' + self.job_label + '] ') if self.job_label else ''}[RUN {run_tag} seed={self.seed} | GPHH evolve] G0 best_fitness={best_fit:.6g}")

        # Track evaluation usage: initial population fully evaluated once
        used = solver.gp_pop * self.eval_budget_per_prog

        # Recompute allowable gens given remaining budget and safe_pop
        cost_per_gen = solver.gp_pop * self.eval_budget_per_prog
        if cost_per_gen <= 0:
            max_gens = 0
        else:
            max_gens = max(0, (gp_budget - used) // cost_per_gen)

        run_gens = int(min(self.gp_gens, max_gens))
        for g in range(1, run_gens + 1):
            new_pop = []
            # Elitism: keep current best
            new_pop.append(best_prog.copy())

            while len(new_pop) < solver.gp_pop:
                a, b = solver._select_parents()
                c, d = solver._crossover(a, b)
                c = solver._mutate(c)
                if len(new_pop) < solver.gp_pop:
                    new_pop.append(c)
                if len(new_pop) < solver.gp_pop:
                    d = solver._mutate(d)
                    new_pop.append(d)

            solver._pop = new_pop
            # Evaluate full population
            solver._fitness = np.array([solver._evaluate_program(p) for p in solver._pop], dtype=float)
            used += solver.gp_pop * self.eval_budget_per_prog
            best_idx = int(np.argmin(solver._fitness))
            if solver._fitness[best_idx] < best_fit:
                best_fit = float(solver._fitness[best_idx])
                best_prog = solver._pop[best_idx]
            if self.verbose:
                print(f"{('[OBJ ' + self.job_label + '] ') if self.job_label else ''}[RUN {run_tag} seed={self.seed} | GPHH evolve] G{g}/{run_gens} used={used}/{gp_budget} best_fitness={best_fit:.6g}")

        desc = _G.GPHH._program_to_str(best_prog)
        return best_prog, desc, used

    # ----------------------- Internal: Stage 3 ----------------------
    def _apply_program_from_incumbent(self, prog: _G.Node, x0: np.ndarray, f0: float, budget: int) -> Result:
        if budget <= 0:
            return Result(x_best=np.asarray(x0), f_best=float(f0), evaluations=0, runtime_sec=0.0, history_best=[float(f0)])

        rng = np.random.default_rng(int(self.rng.integers(0, 10_000_000)))
        lo, hi, D = self.lo, self.hi, self.D

        # Initialize state from incumbent
        state = _G.SearchState(rng=rng, lo=lo, hi=hi, dim=D, x_best=np.asarray(x0).copy(), f_best=float(f0), T=1.0)
        x = np.asarray(x0).copy()
        fx = float(f0)
        best = fx
        x_best = x.copy()

        # Cooling consistent with GPHH
        def _cooling(step: int, B: int, T0: float = 1.0, Tend: float = 1e-3) -> float:
            alpha = (Tend / T0) ** (1.0 / max(1, B - 1))
            return float(T0 * (alpha ** step))

        used = 0
        step = 0
        hist = [best]
        t0 = time.time()
        run_no = (int(self.run_index) + 1) if (self.run_index is not None) else 1
        run_tag = f"{run_no}/{self.run_total}" if (self.run_total is not None and self.run_total > 0) else f"{run_no}"
        while used < budget:
            state.T = _cooling(step, budget)
            # Always use 1-eval steps to realize a proper per-eval annealing schedule.
            # Printing remains controlled by self.verbose.
            step_budget = 1
            x, fx, inc = _G._eval_block(prog, x, fx, state, self.obj, rng, min(step_budget, budget - used))
            used += int(inc)
            step += max(1, int(inc))
            if fx < best:
                best = fx
                x_best = x.copy()
            hist.append(best)
            if self.verbose and (used % max(1, self.print_every) == 0):
                print(f"{('[OBJ ' + self.job_label + '] ') if self.job_label else ''}[RUN {run_tag} seed={self.seed} | APPLY] iter={used}/{budget} | f={fx:.6f} | f_BEST={best:.6f}")
        return Result(x_best=x_best, f_best=float(best), evaluations=used, runtime_sec=float(time.time() - t0), history_best=hist)


__all__ = ["HPHH", "Result"]
