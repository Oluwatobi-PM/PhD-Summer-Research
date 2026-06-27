"""Chapter 4 genetic algorithms translated from the MATLAB implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import CaseConfig
from .variables import (
    binary_indices,
    form_var_ga,
    repair_bounds,
    repair_feasible,
    well_slice,
)


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
    pr: float | None = None
    chrom: np.ndarray | None = None
    ref_chrom: np.ndarray | None = None
    objv: np.ndarray | None = None
    ref_objv: np.ndarray | None = None
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        self.maxgen = self.config.maxgen if self.maxgen is None else self.maxgen
        self.nind = self.config.population_size if self.nind is None else self.nind
        self.pc = self.config.crossover_probability if self.pc is None else self.pc
        self.pm = self.config.mutation_probability if self.pm is None else self.pm
        self.pr = self.config.reference_update_probability if self.pr is None else self.pr
        self.rf = np.arange(1, self.nind + 1, dtype=float)
        self.history_chrom: list[np.ndarray] = []
        self.history_obj: list[np.ndarray] = []


def run_ga(ga: GAData, seed: int = 1000, save_history: bool = True) -> GAData:
    """Run the selected chapter 4 algorithm."""

    algorithm = ga.config.algorithm
    rng = np.random.default_rng(seed if algorithm != "MixencodeGA" else 10000)
    if algorithm == "Iterative":
        return run_iterative(ga, rng, save_history)
    if algorithm == "GenocopIII":
        return run_genocop(ga, rng, save_history)
    return run_mixencode(ga, rng, save_history)


def run_mixencode(ga: GAData, rng: np.random.Generator, save_history: bool) -> GAData:
    if ga.chrom is None:
        ga.chrom = create_combined_population(ga.config, ga.nind, rng)
    for gen in range(int(ga.maxgen)):
        ga.generation = gen
        previous = ga.chrom.copy()
        evolve_combined_generation(ga, rng, repair_mode="heuristic")
        record_generation(ga, previous, "MixencodeGA", save_history)
    print_results(ga)
    return ga


def run_genocop(ga: GAData, rng: np.random.Generator, save_history: bool) -> GAData:
    if ga.chrom is None or ga.ref_chrom is None:
        ga.chrom, ga.ref_chrom = create_genocop_populations(ga.config, ga.nind, rng)
    for gen in range(int(ga.maxgen)):
        ga.generation = gen
        previous = ga.chrom.copy()
        evolve_genocop_generation(ga, rng)
        record_generation(ga, previous, "GenocopIII", save_history)
    print_results(ga)
    return ga


def run_iterative(ga: GAData, rng: np.random.Generator, save_history: bool) -> GAData:
    if ga.chrom is None:
        ga.chrom = create_iterative_population(ga.config, ga.nind, rng)
    for gen in range(int(ga.maxgen)):
        ga.generation = gen
        previous = ga.chrom.copy()
        evolve_iterative_generation(ga, rng)
        record_generation(ga, previous, "Iterative", save_history)
    print_results(ga)
    return ga


def create_combined_population(cfg: CaseConfig, nind: int, rng: np.random.Generator) -> np.ndarray:
    """Create `[active,type,i,j]` chromosomes for MixencodeGA/GenocopIII."""

    pop = np.zeros((nind, cfg.combined_length), dtype=float)
    for i in range(nind):
        pop[i] = random_combined_chromosome(cfg, rng)
    ref = reference_combined_chromosome(cfg)
    for _ in range(min(2, nind)):
        pop[int(rng.integers(0, nind))] = ref
    for _ in range(min(2, nind)):
        pop[int(rng.integers(0, nind))] = random_reference_location_chromosome(cfg, rng)
    return pop


def create_genocop_populations(cfg: CaseConfig, nind: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    search = create_combined_population(cfg, nind, rng)
    ref = np.zeros_like(search)
    for i in range(nind):
        ref[i] = random_combined_chromosome(cfg, rng)
    reference = reference_combined_chromosome(cfg)
    for _ in range(max(1, nind // 2)):
        ref[int(rng.integers(0, nind))] = reference
    return search, ref


def random_combined_chromosome(cfg: CaseConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.locidx1 is not None and cfg.locidx1.shape[0] >= cfg.num_wells:
        loc_rows = cfg.locidx1[rng.choice(cfg.locidx1.shape[0], size=cfg.num_wells, replace=False)]
    else:
        loc_rows = np.column_stack(
            [
                np.arange(1, cfg.num_wells + 1),
                rng.integers(1, cfg.grid_i + 1, size=cfg.num_wells),
                rng.integers(1, cfg.grid_j + 1, size=cfg.num_wells),
            ]
        )
    chrom = np.zeros(cfg.combined_length, dtype=float)
    for idx in range(cfg.num_wells):
        sl = well_slice(cfg, idx)
        chrom[sl.start] = rng.integers(0, 2)
        chrom[sl.start + 1] = 0 if rng.integers(0, 3) < 2 else 1
        chrom[sl.start + 2 : sl.stop] = loc_rows[idx, 1:3]
    return repair_feasible(cfg, chrom)


def reference_combined_chromosome(cfg: CaseConfig) -> np.ndarray:
    if cfg.locidx is None:
        raise RuntimeError("Reference chromosome needs locidx from baseinfo.mat.")
    chrom = np.zeros(cfg.combined_length, dtype=float)
    for idx in range(cfg.num_wells):
        sl = well_slice(cfg, idx)
        chrom[sl.start] = 1
        chrom[sl.start + 1] = 1 if idx < cfg.num_inj else 0
        chrom[sl.start + 2 : sl.stop] = cfg.locidx[idx, 1:3]
    return repair_feasible(cfg, chrom)


def random_reference_location_chromosome(cfg: CaseConfig, rng: np.random.Generator) -> np.ndarray:
    chrom = reference_combined_chromosome(cfg)
    for idx in range(cfg.num_wells):
        sl = well_slice(cfg, idx)
        chrom[sl.start] = rng.integers(0, 2)
        chrom[sl.start + 1] = 0 if rng.integers(0, 3) < 2 else 1
    return chrom


def create_iterative_population(cfg: CaseConfig, nind: int, rng: np.random.Generator) -> np.ndarray:
    """Create `[active,type]` chromosomes for the iterative algorithm."""

    pop = rng.integers(0, 2, size=(nind, cfg.iterative_length)).astype(float)
    for i in range(nind):
        for idx in range(cfg.num_wells):
            if rng.integers(0, 3) < 2:
                pop[i, idx * 2 + 1] = 0
            else:
                pop[i, idx * 2 + 1] = 1
    ref = np.zeros(cfg.iterative_length, dtype=float)
    for idx in range(cfg.num_wells):
        ref[idx * 2] = 1
        ref[idx * 2 + 1] = 1 if idx < cfg.num_inj else 0
    for idx in rng.choice(nind, size=min(2, nind), replace=False):
        pop[idx] = ref
    return pop


def evolve_combined_generation(ga: GAData, rng: np.random.Generator, repair_mode: str) -> None:
    chrom = ga.chrom
    objv = ga.objective(chrom)
    ga.objv = objv
    update_best(ga, chrom, objv)
    fitnv = ranking(objv, ga.rf)
    selected = select_sus(chrom, fitnv, rng)
    crossed = mixed_crossover(selected, ga.config, ga.pc, rng)
    mutated = mutate_combined(crossed, ga.config, ga.pm, rng)
    if repair_mode == "heuristic":
        mutated = np.vstack([repair_feasible(ga.config, row) for row in mutated])
    elite_idx = np.argsort(objv)[:2]
    replace_idx = rng.choice(chrom.shape[0], size=min(2, chrom.shape[0]), replace=False)
    mutated[replace_idx] = chrom[elite_idx[: len(replace_idx)]]
    ga.chrom = mutated
    print_iteration(ga)


def evolve_genocop_generation(ga: GAData, rng: np.random.Generator) -> None:
    if ga.ref_objv is None:
        ga.ref_objv = ga.objective(ga.ref_chrom)
    repaired, refidx = repair_against_reference(ga.config, ga.chrom, ga.ref_chrom, rng)
    objv = ga.objective(repaired)
    ga.objv = objv
    update_best(ga, ga.ref_chrom, ga.ref_objv)
    update_best(ga, repaired, objv)
    for i, ridx in enumerate(refidx):
        if objv[i] < ga.ref_objv[ridx]:
            ga.ref_objv[ridx] = objv[i]
            ga.ref_chrom[ridx] = repaired[i]
        if rng.random() < ga.pr:
            ga.chrom[i] = repaired[i]
    fitnv = ranking(objv, ga.rf)
    selected = select_sus(ga.chrom, fitnv, rng)
    crossed = real_crossover(selected, ga.config, ga.pc, rng)
    mutated = mutate_combined(crossed, ga.config, ga.pm, rng)
    elite_idx = np.argsort(objv)[:2]
    replace_idx = rng.choice(ga.chrom.shape[0], size=min(2, ga.chrom.shape[0]), replace=False)
    mutated[replace_idx] = ga.chrom[elite_idx[: len(replace_idx)]]
    ga.chrom = mutated
    print_iteration(ga)


def evolve_iterative_generation(ga: GAData, rng: np.random.Generator) -> None:
    combined = form_var_ga(ga.config, ga.chrom)
    objv = ga.objective(combined)
    ga.objv = objv
    update_best(ga, ga.chrom, objv)
    fitnv = ranking(objv, ga.rf)
    selected = select_sus(ga.chrom, fitnv, rng)
    crossed = one_point_crossover(selected, ga.pc, rng)
    mutated = mutate_binary(crossed, ga.pm, rng)
    elite_idx = np.argsort(objv)[:2]
    replace_idx = rng.choice(ga.chrom.shape[0], size=min(2, ga.chrom.shape[0]), replace=False)
    mutated[replace_idx] = ga.chrom[elite_idx[: len(replace_idx)]]
    if ga.xmin is not None:
        mutated[round(ga.nind / 2) - 1] = ga.xmin
    ga.chrom = mutated
    print_iteration(ga)


def repair_against_reference(
    cfg: CaseConfig, chrom: np.ndarray, ref_chrom: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    repaired = chrom.copy()
    refidx = rng.integers(0, ref_chrom.shape[0], size=chrom.shape[0])
    for i, ridx in enumerate(refidx):
        candidate = repair_feasible(cfg, repaired[i])
        if not np.array_equal(candidate, repaired[i]):
            alpha = rng.uniform(-0.25, 1.25)
            candidate = repair_bounds(cfg, alpha * repaired[i] + (1 - alpha) * ref_chrom[ridx])
        repaired[i] = repair_feasible(cfg, candidate)
    return repaired, refidx


def update_best(ga: GAData, chrom: np.ndarray, objv: np.ndarray) -> None:
    best_idx = int(np.argmin(objv))
    best_val = float(objv[best_idx])
    if best_val <= ga.fxmin:
        ga.xmin = chrom[best_idx].copy()
        ga.fxmin = best_val


def ranking(objv: np.ndarray, rfun: np.ndarray) -> np.ndarray:
    pos = np.argsort(objv)
    fitv = np.zeros_like(rfun, dtype=float)
    fitv[pos] = rfun[::-1]
    return fitv


def select_sus(chrom: np.ndarray, fitnv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    total = float(np.sum(fitnv))
    step = total / len(fitnv)
    pointer = rng.random() * step
    selected: list[int] = []
    cumulative = 0.0
    for i, fitness in enumerate(fitnv):
        cumulative += float(fitness)
        while cumulative >= pointer and len(selected) < len(fitnv):
            selected.append(i)
            pointer += step
    out = chrom[np.asarray(selected)].copy()
    return out[rng.permutation(out.shape[0])]


def mixed_crossover(pop: np.ndarray, cfg: CaseConfig, pc: float, rng: np.random.Generator) -> np.ndarray:
    out = pop.copy()
    idxb = binary_indices(cfg)
    for i in range(0, pop.shape[0] - 1, 2):
        if rng.random() > pc:
            continue
        a1, a2 = rng.uniform(-0.25, 1.25, size=2)
        c1 = repair_bounds(cfg, a1 * pop[i] + (1 - a1) * pop[i + 1])
        c2 = repair_bounds(cfg, a2 * pop[i] + (1 - a2) * pop[i + 1])
        c1[idxb] = pop[i, idxb]
        c2[idxb] = pop[i + 1, idxb]
        point = int(np.clip(rng.integers(1, len(idxb)), 1, len(idxb) - 1))
        c1b = c1[idxb].copy()
        c2b = c2[idxb].copy()
        c1b[point:], c2b[point:] = c2b[point:].copy(), c1b[point:].copy()
        c1[idxb] = c1b
        c2[idxb] = c2b
        out[i], out[i + 1] = c1, c2
    return out


def real_crossover(pop: np.ndarray, cfg: CaseConfig, pc: float, rng: np.random.Generator) -> np.ndarray:
    out = pop.copy()
    for i in range(0, pop.shape[0] - 1, 2):
        if rng.random() > pc:
            continue
        a1, a2 = rng.uniform(-0.25, 1.25, size=2)
        out[i] = repair_bounds(cfg, a1 * pop[i] + (1 - a1) * pop[i + 1])
        out[i + 1] = repair_bounds(cfg, a2 * pop[i] + (1 - a2) * pop[i + 1])
    return out


def one_point_crossover(pop: np.ndarray, pc: float, rng: np.random.Generator) -> np.ndarray:
    out = pop.copy()
    for i in range(0, pop.shape[0] - 1, 2):
        if rng.random() > pc:
            continue
        point = int(np.clip(rng.integers(1, pop.shape[1]), 1, pop.shape[1] - 1))
        out[i, point:], out[i + 1, point:] = pop[i + 1, point:], pop[i, point:]
    return out


def mutate_combined(pop: np.ndarray, cfg: CaseConfig, pm: float, rng: np.random.Generator) -> np.ndarray:
    out = pop.copy()
    for row in out:
        for idx in range(cfg.num_wells):
            sl = well_slice(cfg, idx)
            if rng.random() < pm:
                row[sl.start] = 1 - row[sl.start]
            if rng.random() < pm:
                row[sl.start + 1] = 1 - row[sl.start + 1]
            if rng.random() < pm:
                row[sl.start + 2] = np.clip(row[sl.start + 2] + rng.choice([-1, 1]), 1, cfg.grid_i)
            if rng.random() < pm:
                row[sl.start + 3] = np.clip(row[sl.start + 3] + rng.choice([-1, 1]), 1, cfg.grid_j)
    return out


def mutate_binary(pop: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    out = pop.copy()
    mask = rng.random(pop.shape) < pm
    out[mask] = 1 - out[mask]
    return out


def record_generation(ga: GAData, previous: np.ndarray, algorithm: str, save_history: bool) -> None:
    ga.xmingen.append(ga.xmin.copy())
    ga.fxmingen.append(float(ga.fxmin))
    if save_history:
        save_generation_data(ga, previous, algorithm)


def save_generation_data(ga: GAData, previous: np.ndarray, algorithm: str) -> None:
    out = Path(ga.config.work_dir) / "python_tempdata"
    out.mkdir(parents=True, exist_ok=True)
    if algorithm == "Iterative":
        history_chrom = form_var_ga(ga.config, previous)
        best = form_var_ga(ga.config, ga.xmin).reshape(-1)
    elif algorithm == "GenocopIII" and ga.ref_chrom is not None:
        history_chrom = np.vstack([ga.ref_chrom, previous])
        best = ga.xmin
    else:
        history_chrom = previous
        best = ga.xmin
    ga.history_chrom.append(history_chrom.copy())
    ga.history_obj.append(ga.objv.copy())
    np.savez_compressed(
        out / "tempdata.npz",
        GAgen=np.array(ga.history_chrom, dtype=object),
        GAgenb=np.array([best for _ in ga.xmingen], dtype=object),
        GAobj=np.array(ga.history_obj, dtype=object),
        GAobjb=-np.array(ga.fxmingen),
    )


def print_iteration(ga: GAData) -> None:
    print("------------------------------------------------")
    print(f"Iteration: {ga.generation}")
    print(f"   xmin: {ga.xmin.tolist()} -- f(xmin): {ga.fxmin}")


def print_results(ga: GAData) -> None:
    print("------------------------------------------------")
    print("######   RESULT   #########")
    print(f"   Objective function for xmin: {ga.fxmin}")
    print(f"   xmin: {ga.xmin.tolist() if ga.xmin is not None else None}")
    print("------------------------------------------------")
