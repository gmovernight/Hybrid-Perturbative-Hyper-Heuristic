"""
Generation Perturbative Hyper-Heuristic (GPHH)
Disposable-only version: Steps 1–4 integrated (no reusable mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Any
import numpy as np
import time

# Robust import of OBJECTIVES
try:
    from benchmark_functions import OBJECTIVES  # name -> (func, lo, hi, dim)
except Exception:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from benchmark_functions import OBJECTIVES


# ------------------------- Public result structure -------------------------

@dataclass
class Result:
    x_best: np.ndarray
    f_best: float
    evaluations: int
    runtime_sec: float
    history_best: List[float]


# ----------------------------- Type aliases --------------------------------

Objective = Callable[[np.ndarray], float]
Bounds = Tuple[np.ndarray, np.ndarray]


# ------------------------------- GPHH API ----------------------------------

class GPHH:
    def __init__(
        self,
        objective: Objective,
        bounds: Bounds,
        dim: int,
        *,
        max_evals: int = 200_000,
        seed: Optional[int] = None,
        gp_pop: int = 100,
        gp_gens: int = 30,
        eval_budget_per_prog: int = 3_000,
        tree_depth_max: int = 5,
        tournament_k: int = 4,
        p_cx: float = 0.8,
        p_mut: float = 0.2,
        verbose: bool = False,
        print_every: int = 1,
    ) -> None:
        # Problem
        self.obj: Objective = objective
        self.lo, self.hi = bounds
        self.D: int = int(dim)
        # Budgets
        self.max_evals = int(max_evals)
        self.eval_budget_per_prog = int(eval_budget_per_prog)
        # GP params
        self.gp_pop = int(gp_pop)
        self.gp_gens = int(gp_gens)
        self.tree_depth_max = int(tree_depth_max)
        self.tournament_k = int(tournament_k)
        self.p_cx = float(p_cx)
        self.p_mut = float(p_mut)
        # Misc
        self.verbose = bool(verbose)
        self.print_every = int(print_every)
        # RNG
        self.rng = np.random.default_rng(seed)

        # Internal
        self._pop: List[Any] = []
        self._fitness: np.ndarray = np.array([])
        self._best_prog: Any = None
        self._best_prog_desc: str = ""

    def run(self) -> Result:
        """Evolve and apply the best program (disposable mode)."""
        return _run(self)

    # Placeholders (bound later)
    def _init_population(self) -> None: ...
    def _select_parents(self): ...
    def _crossover(self, a, b): ...
    def _mutate(self, a): ...
    def _apply_best_program(self, prog): ...
    def _evaluate_program(self, prog): ...


# =============================== Step 2 =====================================
# Primitive operators & state
# ============================================================================

@dataclass
class SearchState:
    rng: np.random.Generator
    lo: np.ndarray
    hi: np.ndarray
    dim: int
    x_best: Optional[np.ndarray] = None
    f_best: Optional[float] = None
    T: float = 1.0

def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def op_gaussian_full(x: np.ndarray, state: SearchState, *, sigma_rel: float = 0.1) -> np.ndarray:
    span = state.hi - state.lo
    sigma = sigma_rel * span
    step = state.rng.normal(0.0, sigma, size=state.dim)
    return _clamp(x + step, state.lo, state.hi)

def op_gaussian_kdims(x: np.ndarray, state: SearchState, *, k: int = 3, sigma_rel: float = 0.15) -> np.ndarray:
    """Gaussian step in k randomly chosen coords; ensure k is int in [1, dim]."""
    k_int = int(max(1, min(state.dim, int(round(k)))))
    idx = state.rng.choice(state.dim, size=k_int, replace=False)
    x_new = x.copy()
    span = state.hi - state.lo
    sigma = sigma_rel * span
    x_new[idx] = x_new[idx] + state.rng.normal(0.0, sigma[idx], size=len(idx))
    return _clamp(x_new, state.lo, state.hi)

def op_cauchy_full(x: np.ndarray, state: SearchState, *, scale_rel: float = 0.05) -> np.ndarray:
    span = state.hi - state.lo
    scale = scale_rel * span
    u = state.rng.random(state.dim) - 0.5
    step = np.tan(np.pi * u) * scale
    return _clamp(x + step, state.lo, state.hi)

def op_random_reset_coord(x: np.ndarray, state: SearchState, *, p: float = 0.1) -> np.ndarray:
    mask = state.rng.random(state.dim) < p
    x_new = x.copy()
    if mask.any():
        x_new[mask] = state.rng.uniform(state.lo[mask], state.hi[mask])
    return x_new

def op_opposition_blend(x: np.ndarray, state: SearchState, *, beta: float = 0.7) -> np.ndarray:
    x_op = state.lo + state.hi - x
    x_new = beta * x + (1.0 - beta) * x_op
    return _clamp(x_new, state.lo, state.hi)

def op_pull_to_best(x: np.ndarray, state: SearchState, *, rate: float = 0.2, jitter_rel: float = 0.01) -> np.ndarray:
    if state.x_best is None:
        return x
    direction = state.x_best - x
    span = state.hi - state.lo
    jitter = state.rng.normal(0.0, jitter_rel * span, size=state.dim)
    x_new = x + rate * direction + jitter
    return _clamp(x_new, state.lo, state.hi)

PRIMITIVE_OPS = {
    "GAUSS_FULL": op_gaussian_full,
    "GAUSS_KDIMS": op_gaussian_kdims,
    "CAUCHY_FULL": op_cauchy_full,
    "RESET_COORD": op_random_reset_coord,
    "OPP_BLEND": op_opposition_blend,
    "PULL_TO_BEST": op_pull_to_best,
}


# =============================== Step 3 =====================================
# AST + random programs + GP ops
# ============================================================================

class Node:
    def copy(self) -> "Node":
        raise NotImplementedError
    def children(self):
        return []
    def replace_child(self, idx: int, new_child: "Node") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} has no children")
    def depth(self) -> int:
        return 1 + (max((c.depth() for c in self.children()), default=0))
    def to_repr(self) -> str:
        raise NotImplementedError

@dataclass
class Apply(Node):
    op_name: str
    params: dict = field(default_factory=dict)
    def copy(self) -> "Apply":
        return Apply(self.op_name, dict(self.params))
    def to_repr(self) -> str:
        if self.params:
            args = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k,v in self.params.items())
            return f"APPLY({self.op_name}({args}))"
        return f"APPLY({self.op_name})"

@dataclass
class Seq(Node):
    stmts: list
    def children(self):
        return list(self.stmts)
    def replace_child(self, idx: int, new_child: Node) -> None:
        self.stmts[idx] = new_child
    def copy(self) -> "Seq":
        return Seq([s.copy() for s in self.stmts])
    def to_repr(self) -> str:
        inner = ", ".join(s.to_repr() for s in self.stmts)
        return f"SEQ({inner})"

@dataclass
class Repeat(Node):
    k: int
    body: Node
    def children(self):
        return [self.body]
    def replace_child(self, idx: int, new_child: Node) -> None:
        assert idx == 0
        self.body = new_child
    def copy(self) -> "Repeat":
        return Repeat(self.k, self.body.copy())
    def to_repr(self) -> str:
        return f"REPEAT({self.k}, {self.body.to_repr()})"

@dataclass
class If(Node):
    cond: dict
    then_block: Node
    else_block: Node
    def children(self):
        return [self.then_block, self.else_block]
    def replace_child(self, idx: int, new_child: Node) -> None:
        if idx == 0:
            self.then_block = new_child
        else:
            self.else_block = new_child
    def copy(self) -> "If":
        return If(dict(self.cond), self.then_block.copy(), self.else_block.copy())
    def to_repr(self) -> str:
        return f"IF({self.cond}) THEN {self.then_block.to_repr()} ELSE {self.else_block.to_repr()}"

def _rand_op_params(rng: np.random.Generator, op_name: str, dim: int) -> dict:
    if op_name == "GAUSS_FULL":
        return {"sigma_rel": float(rng.uniform(0.05, 0.3))}
    if op_name == "GAUSS_KDIMS":
        k = int(rng.integers(low=2, high=max(3, min(6, dim+1))))
        return {"k": int(k), "sigma_rel": float(rng.uniform(0.05, 0.3))}
    if op_name == "CAUCHY_FULL":
        return {"scale_rel": float(rng.uniform(0.01, 0.2))}
    if op_name == "RESET_COORD":
        return {"p": float(rng.uniform(0.05, 0.3))}
    if op_name == "OPP_BLEND":
        return {"beta": float(rng.uniform(0.5, 0.9))}
    if op_name == "PULL_TO_BEST":
        return {"rate": float(rng.uniform(0.05, 0.4)), "jitter_rel": float(rng.uniform(0.001, 0.05))}
    return {}

def _rand_cond(rng: np.random.Generator) -> dict:
    t = rng.choice(["IMPROVES", "RAND_LT", "TEMP_GT"])
    if t == "IMPROVES":
        return {"type": "IMPROVES"}
    elif t == "RAND_LT":
        return {"type": "RAND_LT", "p": float(rng.uniform(0.1, 0.6))}
    else:
        return {"type": "TEMP_GT", "t": float(rng.uniform(0.05, 0.5))}

def _rand_stmt(rng: np.random.Generator, depth: int, depth_max: int, dim: int) -> Node:
    if depth >= depth_max - 1:
        op_name = rng.choice(list(PRIMITIVE_OPS.keys()))
        return Apply(op_name, _rand_op_params(rng, op_name, dim))
    choice = rng.random()
    if choice < 0.5:
        op_name = rng.choice(list(PRIMITIVE_OPS.keys()))
        return Apply(op_name, _rand_op_params(rng, op_name, dim))
    elif choice < 0.7:
        k = int(rng.integers(2, 6))
        body = _rand_block(rng, depth+1, depth_max, dim, max_len=2)
        return Repeat(int(k), body)
    else:
        cond = _rand_cond(rng)
        then_block = _rand_block(rng, depth+1, depth_max, dim, max_len=2)
        else_block = _rand_block(rng, depth+1, depth_max, dim, max_len=2)
        return If(cond, then_block, else_block)

def _rand_block(rng: np.random.Generator, depth: int, depth_max: int, dim: int, max_len: int = 3) -> Node:
    n = int(rng.integers(1, max_len+1))
    stmts = [ _rand_stmt(rng, depth, depth_max, dim) for _ in range(n) ]
    if n == 1:
        return stmts[0]
    return Seq(stmts)

def _random_program(rng: np.random.Generator, depth_max: int, dim: int) -> Node:
    return _rand_block(rng, depth=1, depth_max=depth_max, dim=dim, max_len=3)

def _gather_nodes(root: Node):
    stack = [(None, -1, root)]
    while stack:
        parent, idx, node = stack.pop()
        yield (parent, idx, node)
        for i, c in enumerate(list(node.children())):
            stack.append((node, i, c))

def _clone(root: Node) -> Node:
    return root.copy()

def _random_node_address(rng: np.random.Generator, root: Node):
    nodes = list(_gather_nodes(root))
    return nodes[int(rng.integers(0, len(nodes)))]

def _replace_at(parent: Node, idx: int, new_child: Node, root: Node) -> Node:
    if parent is None:
        return new_child
    parent.replace_child(idx, new_child)
    return root

def _tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int) -> int:
    m = len(fitness)
    idxs = rng.integers(0, m, size=k)
    best = int(idxs[0]); best_fit = fitness[best]
    for j in idxs[1:]:
        j = int(j)
        if fitness[j] < best_fit:
            best, best_fit = j, fitness[j]
    return best

def _crossover_subtree(rng: np.random.Generator, a: Node, b: Node, depth_max: int) -> tuple[Node, Node]:
    a2 = _clone(a); b2 = _clone(b)
    pa, ia, na = _random_node_address(rng, a2)
    pb, ib, nb = _random_node_address(rng, b2)
    if pa is None: new_a = nb.copy()
    else: pa.replace_child(ia, nb.copy()); new_a = a2
    if pb is None: new_b = na.copy()
    else: pb.replace_child(ib, na.copy()); new_b = b2
    if new_a.depth() > depth_max: new_a = a.copy()
    if new_b.depth() > depth_max: new_b = b.copy()
    return new_a, new_b

def _mutate_subtree(rng: np.random.Generator, root: Node, depth_max: int, dim: int) -> Node:
    parent, idx, node = _random_node_address(rng, root)
    new_sub = _random_program(rng, depth_max=depth_max, dim=dim)
    new_root = _replace_at(parent, idx, new_sub, root)
    if new_root.depth() > depth_max:
        return root
    return new_root

def _mutate_point(rng: np.random.Generator, root: Node, dim: int) -> Node:
    parent, idx, node = _random_node_address(rng, root)
    node = node.copy()
    if isinstance(node, Apply):
        params = dict(node.params)
        if node.op_name == "GAUSS_KDIMS":
            k0 = int(round(params.get("k", 3)))
            if rng.random() < 0.6:
                k0 = int(np.clip(k0 + int(rng.integers(-1, 2)), 1, dim))
            params["k"] = int(k0)
        for key, v in list(params.items()):
            if key == "k":
                continue
            if isinstance(v, (float, np.floating, int)) and not isinstance(v, bool):
                vv = float(v)
                params[key] = vv * (1.0 + float(rng.normal(0.0, 0.2)))
        node.params = params
    elif isinstance(node, Repeat):
        node.k = max(1, min(8, int(node.k + int(rng.integers(-1, 2)))))
    elif isinstance(node, If):
        t = node.cond.get("type", "IMPROVES")
        if t == "RAND_LT":
            pval = float(node.cond.get("p", 0.3))
            node.cond["p"] = float(np.clip(pval * (1.0 + rng.normal(0.0, 0.2)), 0.01, 0.99))
        elif t == "TEMP_GT":
            tval = float(node.cond.get("t", 0.2))
            node.cond["t"] = float(np.clip(tval * (1.0 + rng.normal(0.0, 0.2)), 1e-3, 10.0))
    if parent is None:
        return node
    parent = parent.copy()
    parent.replace_child(idx, node)
    return root

def _program_to_str(root: Node) -> str:
    return root.to_repr()

def _gphh_init_population(self):
    self._pop = [_random_program(self.rng, self.tree_depth_max, self.D) for _ in range(self.gp_pop)]
    self._fitness = np.full(self.gp_pop, np.inf, dtype=float)

def _gphh_select_parents(self):
    i = _tournament_select(self.rng, self._fitness, self.tournament_k)
    j = _tournament_select(self.rng, self._fitness, self.tournament_k)
    while j == i and self.gp_pop > 1:
        j = _tournament_select(self.rng, self._fitness, self.tournament_k)
    return self._pop[i], self._pop[j]

def _gphh_crossover(self, a, b):
    if self.rng.random() < self.p_cx:
        return _crossover_subtree(self.rng, a, b, self.tree_depth_max)
    return a.copy(), b.copy()

def _gphh_mutate(self, a):
    if self.rng.random() < 0.5:
        return _mutate_subtree(self.rng, a, self.tree_depth_max, self.D)
    else:
        return _mutate_point(self.rng, a, self.D)

# Bind Step 3 ops
GPHH._init_population = _gphh_init_population
GPHH._select_parents = _gphh_select_parents
GPHH._crossover = _gphh_crossover
GPHH._mutate = _gphh_mutate
GPHH._program_to_str = staticmethod(_program_to_str)


# =============================== Step 4 =====================================
# Interpreter, evaluation, and main loop
# ============================================================================

def _propose_and_accept(rng: np.random.Generator, op_func, params: dict,
                        x: np.ndarray, fx: float, state: SearchState, fobj: Objective):
    x_prop = op_func(x, state, **(params or {}))
    fx_prop = fobj(x_prop)
    if fx_prop < fx:
        x_new, fx_new = x_prop, fx_prop
    else:
        delta = float(fx_prop - fx)
        T = max(1e-12, float(state.T))
        p = float(np.exp(-delta / T))
        if rng.random() < p:
            x_new, fx_new = x_prop, fx_prop
        else:
            x_new, fx_new = x, fx
    if state.f_best is None or fx_prop < state.f_best:
        state.f_best = float(fx_prop)
        state.x_best = x_prop.copy()
    return x_new, fx_new, 1  # 1 eval used

def _eval_block(node: Node, x: np.ndarray, fx: float, state: SearchState,
                fobj: Objective, rng: np.random.Generator, budget: int):
    if budget <= 0:
        return x, fx, 0

    if isinstance(node, Apply):
        op = PRIMITIVE_OPS[node.op_name]
        x_new, fx_new, used = _propose_and_accept(rng, op, node.params, x, fx, state, fobj)
        return x_new, fx_new, used

    elif isinstance(node, Seq):
        used_total = 0
        for s in node.stmts:
            x, fx, used = _eval_block(s, x, fx, state, fobj, rng, budget - used_total)
            used_total += used
            if used_total >= budget:
                break
        return x, fx, used_total

    elif isinstance(node, Repeat):
        used_total = 0
        k = max(1, int(node.k))
        for _ in range(k):
            if used_total >= budget:
                break
            x, fx, used = _eval_block(node.body, x, fx, state, fobj, rng, budget - used_total)
            used_total += used
        return x, fx, used_total

    elif isinstance(node, If):
        t = node.cond.get("type", "IMPROVES")
        if t == "RAND_LT":
            p = float(node.cond.get("p", 0.3))
            branch = node.then_block if rng.random() < p else node.else_block
            return _eval_block(branch, x, fx, state, fobj, rng, budget)
        elif t == "TEMP_GT":
            thr = float(node.cond.get("t", 0.2))
            branch = node.then_block if float(state.T) > thr else node.else_block
            return _eval_block(branch, x, fx, state, fobj, rng, budget)
        else:  # IMPROVES
            x_then, fx_then, used_then = _eval_block(node.then_block, x, fx, state, fobj, rng, budget)
            if fx_then < fx:
                return x_then, fx_then, used_then
            used_total = used_then
            if used_total >= budget:
                return x, fx, used_total
            x_else, fx_else, used_else = _eval_block(node.else_block, x, fx, state, fobj, rng, budget - used_total)
            used_total += used_else
            if fx_else < fx:
                return x_else, fx_else, used_total
            else:
                return x, fx, used_total

    else:
        return x, fx, 0

def _cooling_schedule(step: int, total_steps: int, T0: float = 1.0, Tend: float = 1e-3) -> float:
    if total_steps <= 1:
        return Tend
    alpha = (Tend / T0) ** (1.0 / max(1, total_steps-1))
    return float(T0 * (alpha ** step))

def _evaluate_program(self, prog: Node) -> float:
    fobj = self.obj
    rng = self.rng
    lo, hi, D = self.lo, self.hi, self.D
    budget = self.eval_budget_per_prog

    x = rng.uniform(lo, hi)
    fx = float(fobj(x))
    best = fx
    state = SearchState(rng=rng, lo=lo, hi=hi, dim=D, x_best=x.copy(), f_best=fx, T=1.0)

    used = 0; step = 0
    while used < budget:
        state.T = _cooling_schedule(step, budget, T0=1.0, Tend=1e-3)
        x, fx, inc = _eval_block(prog, x, fx, state, fobj, rng, budget - used)
        used += inc; step += max(1, inc)
        if fx < best:
            best = fx
    return float(best)

def _apply_best_program(self, prog: Node) -> Result:
    fobj = self.obj
    rng = self.rng
    lo, hi, D = self.lo, self.hi, self.D
    budget = self.max_evals

    x = rng.uniform(lo, hi)
    fx = float(fobj(x))
    best = fx; x_best = x.copy(); hist = [best]
    state = SearchState(rng=rng, lo=lo, hi=hi, dim=D, x_best=x_best.copy(), f_best=best, T=1.0)

    used = 0; step = 0
    t0 = time.time()
    while used < budget:
        state.T = _cooling_schedule(step, budget, T0=1.0, Tend=1e-3)
        x, fx, inc = _eval_block(prog, x, fx, state, fobj, rng, budget - used)
        used += inc; step += max(1, inc)
        if fx < best:
            best = fx; x_best = x.copy()
        hist.append(best)
    elapsed = time.time() - t0
    return Result(x_best=x_best, f_best=float(best), evaluations=used, runtime_sec=float(elapsed), history_best=hist)

def _run(self) -> Result:
    t0 = time.time()
    self._init_population()

    # Initial eval
    self._fitness = np.array([_evaluate_program(self, p) for p in self._pop], dtype=float)
    best_idx = int(np.argmin(self._fitness))
    best_prog = self._pop[best_idx]
    best_fit = float(self._fitness[best_idx])
    if self.verbose:
        print(f"[G0] best fitness = {best_fit:.6g} :: {GPHH._program_to_str(best_prog)}")

    for g in range(1, self.gp_gens + 1):
        new_pop = []
        elite = best_prog.copy()
        new_pop.append(elite)

        while len(new_pop) < self.gp_pop:
            a, b = self._select_parents()
            c, d = self._crossover(a, b)
            c = self._mutate(c)
            if len(new_pop) < self.gp_pop:
                new_pop.append(c)
            if len(new_pop) < self.gp_pop:
                d = self._mutate(d)
                new_pop.append(d)

        self._pop = new_pop
        self._fitness = np.array([_evaluate_program(self, p) for p in self._pop], dtype=float)
        best_idx = int(np.argmin(self._fitness))
        if self._fitness[best_idx] < best_fit:
            best_fit = float(self._fitness[best_idx])
            best_prog = self._pop[best_idx]
        if self.verbose and (g % self.print_every == 0):
            print(f"[G{g}] best fitness = {best_fit:.6g}")

    self._best_prog = best_prog
    self._best_prog_desc = GPHH._program_to_str(best_prog)
    res = _apply_best_program(self, best_prog)
    res.runtime_sec = float(time.time() - t0)
    return res

# Bind Step 3/4 methods
GPHH._init_population = _gphh_init_population
GPHH._select_parents = _gphh_select_parents
GPHH._crossover = _gphh_crossover
GPHH._mutate = _gphh_mutate
GPHH._evaluate_program = _evaluate_program
GPHH._apply_best_program = _apply_best_program
GPHH._program_to_str = staticmethod(_program_to_str)
GPHH.run = _run

# ============================ Replay & Parser Helpers ============================
# Parse the textual program format produced by _program_to_str (APPLY/SEQ/REPEAT/IF(...)).
# This enables re-applying stored programs to trace convergence without re-evolving.

import ast

def parse_program(prog_str: str) -> Node:
    s = prog_str.strip()

    def skip_ws(i):
        while i < len(s) and s[i].isspace():
            i += 1
        return i

    def expect(i, token):
        i = skip_ws(i)
        if s.startswith(token, i):
            return i + len(token)
        raise ValueError(f"Expected '{token}' at pos {i}: ...{s[max(0,i-15):i+15]}...")

    def parse_number(tok):
        # try int, then float
        try:
            return int(tok)
        except ValueError:
            return float(tok)

    def parse_params(i):
        # parse inside NAME(arg1=val1, arg2=val2)
        # returns (dict, new_i), assumes current s[i] is at start of params (after '(')
        params = {}
        i = skip_ws(i)
        if s[i] == ')':
            return params, i
        while True:
            i = skip_ws(i)
            # read key
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] in "_"):
                j += 1
            key = s[i:j]
            i = skip_ws(j)
            i = expect(i, "=")
            i = skip_ws(i)
            # read value: could be float/int/bool
            # value ends at ',' or ')' (no nested structures here)
            j = i
            while j < len(s) and s[j] not in ",)":
                j += 1
            val_tok = s[i:j].strip()
            # try literal eval (supports floats/ints/bools)
            try:
                val = ast.literal_eval(val_tok)
            except Exception:
                # fallback to number parse
                val = parse_number(val_tok)
            params[key] = val
            i = skip_ws(j)
            if i < len(s) and s[i] == ',':
                i += 1
                continue
            else:
                return params, i

    def parse_apply(i):
        i = expect(i, "APPLY")
        i = expect(i, "(")
        i = skip_ws(i)
        # read op name
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] in "_"):
            j += 1
        op_name = s[i:j]
        i = skip_ws(j)
        params = {}
        if s[i] == '(':
            i += 1
            params, i = parse_params(i)
            i = expect(i, ")")
        i = skip_ws(i)
        i = expect(i, ")")
        return Apply(op_name=op_name, params=params), i

    def parse_seq(i):
        i = expect(i, "SEQ")
        i = expect(i, "(")
        items = []
        while True:
            node, i = parse_block(i)
            items.append(node)
            i = skip_ws(i)
            if s[i] == ',':
                i += 1
                continue
            elif s[i] == ')':
                i += 1
                break
            else:
                raise ValueError(f"SEQ: expected ',' or ')' at pos {i}")
        if len(items) == 1:
            return items[0], i
        return Seq(items), i

    def parse_repeat(i):
        i = expect(i, "REPEAT")
        i = expect(i, "(")
        i = skip_ws(i)
        # parse integer k
        j = i
        while j < len(s) and (s[j].isdigit()):
            j += 1
        k = int(s[i:j])
        i = skip_ws(j)
        i = expect(i, ",")
        body, i = parse_block(i)
        i = skip_ws(i)
        i = expect(i, ")")
        return Repeat(k=k, body=body), i

    def parse_if(i):
        i = expect(i, "IF")
        i = expect(i, "(")
        # read dict literal until matching ')'
        depth = 1
        j = i
        while j < len(s) and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        dict_text = s[i:j].strip()
        try:
            cond = ast.literal_eval(dict_text)
        except Exception as e:
            raise ValueError(f"Could not parse IF condition dict: {dict_text}") from e
        i = j
        i = expect(i, ")")
        i = skip_ws(i)
        i = skip_ws(i)
        if s.startswith("THEN", i):
            i += 4
        else:
            raise ValueError(f"Expected THEN at pos {i}: ...{s[max(0,i-15):i+15]}...")
        then_block, i = parse_block(i)
        i = skip_ws(i)
        if s.startswith("ELSE", i):
            i += 4
        else:
            raise ValueError(f"Expected ELSE at pos {i}: ...{s[max(0,i-15):i+15]}...")
        else_block, i = parse_block(i)
        return If(cond=cond, then_block=then_block, else_block=else_block), i

    def parse_block(i):
        i = skip_ws(i)
        if s.startswith("APPLY", i):
            return parse_apply(i)
        if s.startswith("SEQ", i):
            return parse_seq(i)
        if s.startswith("REPEAT", i):
            return parse_repeat(i)
        if s.startswith("IF", i):
            return parse_if(i)
        raise ValueError(f"Unknown construct at pos {i}: ...{s[max(0,i-15):i+15]}...")

    node, i2 = parse_block(0)
    i2 = skip_ws(i2)
    if i2 != len(s):
        # trailing chars; try to be tolerant
        pass
    return node

def apply_program_from_string(prog_desc, f, bounds, D, seed, evals, trace_every=100):
    #\"\"\"Re-apply a serialized program string with tracing.
    #Returns (evals_list, bestf_list).\"\"\"
    import numpy as _np
    rng = _np.random.default_rng(seed)
    lo, hi = bounds

    # parse program string into AST
    prog = parse_program(str(prog_desc))

    # init state
    state = SearchState(rng=rng, lo=_np.asarray(lo), hi=_np.asarray(hi), dim=int(D))
    x = rng.uniform(state.lo, state.hi)
    fx = float(f(x))
    state.x_best = x.copy()
    state.f_best = fx

    T0, Tend = 1.0, 1e-3
    B = int(evals)
    alpha = (Tend / T0) ** (1.0 / max(1, B - 1))

    evals_list = [0]
    bestf_list = [fx]
    used = 0
    step = 0

    while used < B:
        state.T = float(T0 * (alpha ** step))
        x, fx, inc = _eval_block(prog, x, fx, state, f, rng, B - used)
        used += int(inc)
        step += max(1, int(inc))
        if fx < state.f_best:
            state.f_best = fx
            state.x_best = x.copy()
        if (used % max(1, int(trace_every))) == 0 or used >= B:
            evals_list.append(used)
            bestf_list.append(state.f_best)

    return evals_list, bestf_list
# ========================== End Replay & Parser Helpers ==========================


# ------------------------------- Quick demo --------------------------------
if __name__ == "__main__":
    name = "f13_D10"
    f, lo, hi, D = OBJECTIVES[name]
    solver = GPHH(f, (lo, hi), D, verbose=True)
    print(f"GPHH ready for objective '{name}' with D={D}. Run demo_gphh.py for a full run.")
