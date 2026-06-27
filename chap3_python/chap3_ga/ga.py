"""Genetic algorithm translated from `GA_opt.m`.

The chromosome can contain three logical blocks depending on `design_var`:
drilling order `O`, well type `T`, and binary encoded location `x`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import CaseConfig
from .encoding import bi2de, de2bi, decode_locations, encode_locations
from .initial_solutions import apply_initial_chromosomes, describe_initial_chromosomes


Objective = Callable[[np.ndarray], np.ndarray]


@dataclass
class GAData:
    """Mutable GA state, equivalent to the MATLAB `gaDat` structure."""

    config: CaseConfig
    objective: Objective
    maxgen: int | None = None
    nind: int | None = None
    pc: float | None = None
    pm: float | None = None
    chrom: np.ndarray | None = None
    initial_chromosomes: np.ndarray | None = None
    objv: np.ndarray | None = None
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        # Fill optional parameters from the case configuration, matching
        # `GA_base_struct.m`.
        self.maxgen = self.config.maxgen if self.maxgen is None else self.maxgen
        self.nind = self.config.population_size if self.nind is None else self.nind
        self.pc = self.config.crossover_probability if self.pc is None else self.pc
        self.pm = self.config.mutation_probability if self.pm is None else self.pm
        self.rf = np.arange(1, self.nind + 1, dtype=float)
        self.mutsure = np.zeros(self.nind, dtype=bool)
        self.history_chrom: list[np.ndarray] = []
        self.history_obj: list[np.ndarray] = []


def run_ga(
    ga: GAData,
    seed: int = 1000,
    save_history: bool = True,
    generation_offset: int = 0,
) -> GAData:
    """Run the full GA loop and return the final state."""

    rng = np.random.default_rng(seed)
    if ga.chrom is None:
        ga.chrom = create_population(ga.config, ga.nind, rng)
        seeded_count = apply_initial_chromosomes(ga.chrom, ga.initial_chromosomes)
        if seeded_count:
            print(f"Seeded initial population with {seeded_count} user chromosome(s).", flush=True)
            print(describe_initial_chromosomes(ga.chrom[:seeded_count], ga.config), flush=True)
    for gen in range(int(ga.maxgen)):
        ga.generation = generation_offset + gen
        previous = ga.chrom.copy()
        evolve_generation(ga, rng)
        ga.xmingen.append(ga.xmin.copy())
        ga.fxmingen.append(float(ga.fxmin))
        if save_history:
            save_generation_data(ga, previous)
        print_iteration(ga)
    print_results(ga)
    return ga


def create_population(cfg: CaseConfig, nind: int, rng: np.random.Generator) -> np.ndarray:
    """Create the initial random population (`crtrp` in MATLAB)."""

    chrom = np.zeros((nind, cfg.chromosome_length), dtype=int)
    bpw = cfg.bits_per_location
    for i in range(nind):
        if cfg.design_var in (1, 3):
            # Order genes are permutations of well numbers, kept 1-based to
            # match the MATLAB chromosome and schedule formulas.
            chrom[i, : cfg.num_wells] = rng.permutation(cfg.num_wells) + 1
        if cfg.design_var in (1, 2):
            # Type genes are 0/1; location genes are binary encodings of
            # unique 1-based candidate locations.
            type_start = cfg.beforetype * cfg.num_wells
            loc_start = cfg.beforeloc * cfg.num_wells
            chrom[i, type_start:loc_start] = rng.integers(0, 2, size=cfg.num_wells)
            locations = rng.permutation(cfg.num_locations)[: cfg.num_wells] + 1
            encode_locations(locations, cfg.beforeloc, cfg.num_wells, bpw, chrom[i])
    return chrom


def evolve_generation(ga: GAData, rng: np.random.Generator) -> None:
    """Evaluate, select, cross over, mutate, and keep two elite solutions."""

    chrom = ga.chrom
    objv = ga.objective(chrom)
    ga.objv = objv
    best_idx = int(np.argmin(objv))
    best_val = float(objv[best_idx])
    if best_val <= ga.fxmin:
        ga.xmin = chrom[best_idx].copy()
        ga.fxmin = best_val
    fitnv = ranking(objv, ga.rf)
    selected = select_sus(chrom, fitnv, rng)
    crossed = crossover_population(selected, ga, rng)
    mutated = mutate_population(crossed, ga, rng)
    elite_idx = np.argsort(objv)[:2]
    replace_idx = rng.integers(0, chrom.shape[0], size=2)
    mutated[replace_idx, :] = chrom[elite_idx, :]
    ga.chrom = mutated


def ranking(objv: np.ndarray, rfun: np.ndarray) -> np.ndarray:
    """Rank lower objective values as fitter, mirroring MATLAB `ranking`."""

    pos = np.argsort(objv)
    fitv = np.zeros_like(rfun, dtype=float)
    fitv[pos] = rfun[::-1]
    return fitv


def select_sus(chrom: np.ndarray, fitnv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic universal sampling selection."""

    nsel = len(fitnv)
    total = float(np.sum(fitnv))
    step = total / nsel
    pointer = rng.random() * step
    indices: list[int] = []
    cumulative = 0.0
    for i, fitness in enumerate(fitnv):
        cumulative += float(fitness)
        while cumulative >= pointer and len(indices) < nsel:
            indices.append(i)
            pointer += step
    selected = chrom[np.array(indices), :].copy()
    return selected[rng.permutation(nsel), :]


def crossover_population(old: np.ndarray, ga: GAData, rng: np.random.Generator) -> np.ndarray:
    """Apply mixed-encoding crossover to paired parents."""

    cfg = ga.config
    new = old.copy()
    n_pairs = old.shape[0] // 2
    for pair in range(n_pairs):
        pin = pair * 2
        if rng.random() > ga.pc:
            continue
        p1 = old[pin].copy()
        p2 = old[pin + 1].copy()
        if cfg.design_var == 3:
            c1_order, c2_order = order_crossover(p1[: cfg.num_wells], p2[: cfg.num_wells], cfg.num_wells, rng)
            p1[: cfg.num_wells] = c1_order
            p2[: cfg.num_wells] = c2_order
            new[pin], new[pin + 1] = p1, p2
            continue
        c1, c2 = crossover_pair(p1, p2, cfg, rng.integers(1, old.shape[1]), rng)
        if cfg.design_var in (1, 2):
            c1 = matlab_mutfeasi(c1, cfg, rng)
            c2 = matlab_mutfeasi(c2, cfg, rng)
        new[pin], new[pin + 1] = c1, c2
    return new


def crossover_pair(
    p1: np.ndarray, p2: np.ndarray, cfg: CaseConfig, point: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Cross one pair, using order crossover when the cut lands in `O`."""

    c1 = p1.copy()
    c2 = p2.copy()
    if cfg.design_var == 1 and point <= cfg.num_wells:
        c1o, c2o = order_crossover(p1[: cfg.num_wells], p2[: cfg.num_wells], cfg.num_wells, rng)
        c1[: cfg.num_wells] = c1o
        c2[: cfg.num_wells] = c2o
        c1[cfg.num_wells :] = p2[cfg.num_wells :]
        c2[cfg.num_wells :] = p1[cfg.num_wells :]
    else:
        c1[point:] = p2[point:]
        c2[point:] = p1[point:]
    return c1, c2


def order_crossover(g1: np.ndarray, g2: np.ndarray, num_wells: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Partially mapped crossover for permutation/order genes."""

    n1, n2 = sorted(rng.integers(0, num_wells, size=2))
    if n1 == n2:
        return g1.copy(), g2.copy()
    child1 = np.zeros(num_wells, dtype=int)
    child2 = np.zeros(num_wells, dtype=int)
    child1[n1 : n2 + 1] = g2[n1 : n2 + 1]
    child2[n1 : n2 + 1] = g1[n1 : n2 + 1]
    fill_order_child(child1, g1)
    fill_order_child(child2, g2)
    return child1, child2


def fill_order_child(child: np.ndarray, parent: np.ndarray) -> None:
    for gene in parent:
        if gene not in child:
            child[np.where(child == 0)[0][0]] = gene


def mutate_population(old: np.ndarray, ga: GAData, rng: np.random.Generator) -> np.ndarray:
    """Apply MATLAB-style mutation matrix and feasibility handling."""

    cfg = ga.config
    aux = rng.random(old.shape)
    if cfg.design_var == 2:
        mutmx = aux < ga.pm
    else:
        order_pm = min(cfg.order_mutation_probability * 3.0, 0.3)
        threshold = np.concatenate(
            [np.full(cfg.num_wells, order_pm), np.full(old.shape[1] - cfg.num_wells, ga.pm)]
        )
        mutmx = aux < threshold
    new = old.copy()
    for i in range(old.shape[0]):
        if np.any(mutmx[i]):
            new[i] = mutate_chromosome(old[i], mutmx[i], cfg, rng)
            if cfg.design_var in (1, 2):
                new[i] = matlab_mutfeasi(new[i], cfg, rng)
    return new


def mutate_chromosome(chrom: np.ndarray, idx: np.ndarray, cfg: CaseConfig, rng: np.random.Generator) -> np.ndarray:
    """Mutate one mixed-encoding chromosome."""

    mutated = chrom.copy()
    if cfg.design_var in (1, 3) and np.any(idx[: cfg.num_wells]):
        mutated[: cfg.num_wells] = order_mutation(chrom[: cfg.num_wells], rng)
    if cfg.design_var in (1, 2):
        flip_idx = np.where(idx[cfg.num_wells :])[0] + cfg.num_wells
        mutated[flip_idx] = 1 - mutated[flip_idx]
    return mutated


def order_mutation(order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly swap, rotate, or reverse part of the drilling order."""

    cm = order.copy()
    operator = int(rng.integers(1, 4))
    n = len(order)
    if operator == 1:
        a, b = rng.integers(0, n, size=2)
        cm[a], cm[b] = cm[b], cm[a]
    elif operator == 2:
        cp = int(rng.integers(1, n))
        cm = np.concatenate([order[cp:], order[:cp]])
    else:
        a, b = sorted(rng.integers(0, n, size=2))
        cm[a : b + 1] = cm[a : b + 1][::-1]
    return cm


def matlab_mutfeasi(chrom: np.ndarray, cfg: CaseConfig, rng: np.random.Generator) -> np.ndarray:
    """Python equivalent of MATLAB `mutfeasi`: keep location genes feasible."""

    fixed = chrom.copy()
    bpw = cfg.bits_per_location
    locs = decode_locations(fixed, cfg.beforeloc, cfg.num_wells, bpw)
    available = list(range(1, cfg.num_locations + 1))
    seen: set[int] = set()
    for i, loc in enumerate(locs):
        if loc < 1 or loc > cfg.num_locations or loc in seen:
            choices = [x for x in available if x not in seen and x not in locs[i + 1 :]]
            if not choices:
                choices = [x for x in available if x not in seen]
            locs[i] = int(rng.choice(choices))
        seen.add(int(locs[i]))
    encode_locations(locs, cfg.beforeloc, cfg.num_wells, bpw, fixed)
    return fixed


def print_iteration(ga: GAData) -> None:
    print("------------------------------------------------", flush=True)
    print(f"Iteration: {ga.generation}", flush=True)
    print(f"   xmin: {ga.xmin.tolist()} -- f(xmin): {ga.fxmin}", flush=True)


def print_results(ga: GAData) -> None:
    print("------------------------------------------------", flush=True)
    print("######   RESULT   #########", flush=True)
    print(f"   Objective function for xmin: {ga.fxmin}", flush=True)
    print(f"   xmin: {ga.xmin.tolist() if ga.xmin is not None else None}", flush=True)
    print("------------------------------------------------", flush=True)


def save_generation_data(ga: GAData, previous_chrom: np.ndarray) -> None:
    """Save MATLAB-like generation history in `python_tempdata/tempdata.npz`."""

    out = Path(ga.config.work_dir) / "python_tempdata"
    out.mkdir(parents=True, exist_ok=True)
    ga.history_chrom.append(previous_chrom.copy())
    ga.history_obj.append(ga.objv.copy())
    payload = {
        "GAgen": np.array(ga.history_chrom),
        "GAgenb": np.array(ga.xmingen),
        "GAobj": np.array(ga.history_obj),
        "GAobjb": -np.array(ga.fxmingen),
    }
    target = out / "tempdata.npz"
    tmp = out / "tempdata.tmp.npz"
    try:
        np.savez_compressed(tmp, **payload)
        tmp.replace(target)
    except PermissionError:
        if tmp.exists():
            tmp.unlink()
        raise PermissionError(
            f"Could not update {target}. Close any program that has this file open "
            "and run the optimizer again."
        )
