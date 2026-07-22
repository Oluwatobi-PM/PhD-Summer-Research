"""Differential Evolution for Chapter 3 design variables.

DE evolves continuous vectors in normalized [0, 1] space, then decodes each
vector through the same mixed-integer chromosome pipeline used by ILHS/PSO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from chap3_ga.checkpoint import atomic_savez, best_population_payload, common_metadata, optimizer_history_dir
from chap3_ga.config import CaseConfig
from chap3_ga.lhs_initialization import decode_lhs_population, lhs_population, normalized_dimension


Objective = Callable[[np.ndarray], np.ndarray]


STRATEGY_NAMES = {
    1: "DE/best/1/exp",
    2: "DE/rand/1/exp",
    3: "DE/rand-to-best/1/exp",
    4: "DE/best/2/exp",
    5: "DE/rand/2/exp",
    6: "DE/best/1/bin",
    7: "DE/rand/1/bin",
    8: "DE/rand-to-best/1/bin",
    9: "DE/best/2/bin",
    10: "DE/rand/2/bin",
}


@dataclass
class DEData:
    """Mutable Differential Evolution state."""

    config: CaseConfig
    objective: Objective
    max_generations: int | None = None
    population_size: int | None = None
    mutation_factor: float = 0.7
    crossover_factor: float = 0.5
    strategy: int = 7
    initial_population: np.ndarray | None = None
    population: np.ndarray | None = None
    trial_population: np.ndarray | None = None
    chrom: np.ndarray | None = None
    trial_chrom: np.ndarray | None = None
    objv: np.ndarray | None = None
    trial_objv: np.ndarray | None = None
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    best_particle: np.ndarray | None = None
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    history_population: list[np.ndarray] = field(default_factory=list)
    history_trial_population: list[np.ndarray] = field(default_factory=list)
    history_chrom: list[np.ndarray] = field(default_factory=list)
    history_trial_chrom: list[np.ndarray] = field(default_factory=list)
    history_obj: list[np.ndarray] = field(default_factory=list)
    history_trial_obj: list[np.ndarray] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        self.max_generations = self.config.maxgen if self.max_generations is None else self.max_generations
        self.population_size = self.config.population_size if self.population_size is None else self.population_size
        self.dimension = normalized_dimension(self.config)
        if not 0.0 <= self.crossover_factor <= 1.0:
            raise ValueError("crossover_factor must be between 0 and 1.")
        if self.mutation_factor < 0.0:
            raise ValueError("mutation_factor must be non-negative.")
        if self.strategy not in STRATEGY_NAMES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGY_NAMES)}.")
        min_population = 6 if self.strategy in (4, 5, 9, 10) else 4
        if int(self.population_size) < min_population:
            raise ValueError(f"population_size must be at least {min_population} for {STRATEGY_NAMES[self.strategy]}.")


def run_de(
    de: DEData,
    seed: int = 1000,
    save_history: bool = True,
    generation_offset: int = 0,
    evaluate_initial: bool = True,
) -> DEData:
    """Run Differential Evolution and return the final state."""

    rng = np.random.default_rng(seed)
    population = de.population.copy() if de.population is not None else initial_population(de, rng)
    if evaluate_initial or de.objv is None:
        de.generation = generation_offset
        de.population = population.copy()
        de.chrom = decode_lhs_population(de.config, population)
        de.objv = np.asarray(de.objective(de.chrom), dtype=float)
        update_best(de, population, de.chrom, de.objv)
        de.xmingen.append(de.xmin.copy())
        de.fxmingen.append(float(de.fxmin))
        if save_history:
            de.trial_population = population.copy()
            de.trial_chrom = de.chrom.copy()
            de.trial_objv = de.objv.copy()
            save_generation_data(de)
        print_generation(de)

    objv = np.asarray(de.objv, dtype=float).copy()
    chrom = np.asarray(de.chrom, dtype=int).copy()

    for gen in range(1, int(de.max_generations) + 1):
        de.generation = generation_offset + gen
        trial_population = create_trial_population(de, population, rng)
        trial_chrom = decode_lhs_population(de.config, trial_population)
        trial_objv = np.asarray(de.objective(trial_chrom), dtype=float)

        improved = trial_objv <= objv
        population[improved] = trial_population[improved]
        chrom[improved] = trial_chrom[improved]
        objv[improved] = trial_objv[improved]

        de.population = population.copy()
        de.trial_population = trial_population.copy()
        de.chrom = chrom.copy()
        de.trial_chrom = trial_chrom.copy()
        de.objv = objv.copy()
        de.trial_objv = trial_objv.copy()
        update_best(de, population, chrom, objv)
        de.xmingen.append(de.xmin.copy())
        de.fxmingen.append(float(de.fxmin))
        if save_history:
            save_generation_data(de)
        print_generation(de)

    print_results(de)
    return de


def initial_population(de: DEData, rng: np.random.Generator) -> np.ndarray:
    """Return the initial normalized DE population."""

    if de.initial_population is not None:
        population = np.asarray(de.initial_population, dtype=float).copy()
    else:
        population, _ = lhs_population(int(de.population_size), de.dimension, rng)
    expected = (int(de.population_size), de.dimension)
    if population.shape != expected:
        raise ValueError(f"Initial DE population has shape {population.shape}; expected {expected}.")
    return np.clip(population, 0.0, 1.0)


def create_trial_population(de: DEData, population: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create one DE trial population for the selected strategy."""

    trial = np.zeros_like(population)
    for idx in range(population.shape[0]):
        mutant = mutation_vector(de, population, idx, rng)
        if de.strategy <= 5:
            trial[idx] = exponential_crossover(population[idx], mutant, float(de.crossover_factor), rng)
        else:
            trial[idx] = binomial_crossover(population[idx], mutant, float(de.crossover_factor), rng)
    return np.clip(trial, 0.0, 1.0)


def mutation_vector(de: DEData, population: np.ndarray, target_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Return the mutant vector for one target individual."""

    f = float(de.mutation_factor)
    needed = 5 if de.strategy in (4, 5, 9, 10) else 3
    indices = random_distinct_indices(population.shape[0], target_idx, needed, rng)
    r1, r2, r3 = indices[:3]
    r4 = indices[3] if needed > 3 else r3
    r5 = indices[4] if needed > 4 else r3
    target = population[target_idx]
    if de.best_particle is None:
        best = population[int(np.nanargmin(de.objv))]
    else:
        best = de.best_particle

    if de.strategy in (1, 6):
        mutant = best + f * (population[r2] - population[r3])
    elif de.strategy in (2, 7):
        mutant = population[r1] + f * (population[r2] - population[r3])
    elif de.strategy in (3, 8):
        mutant = target + f * (best - target) + f * (population[r1] - population[r2])
    elif de.strategy in (4, 9):
        mutant = best + f * (population[r1] + population[r2] - population[r3] - population[r4])
    else:
        mutant = population[r5] + f * (population[r1] + population[r2] - population[r3] - population[r4])
    return np.clip(mutant, 0.0, 1.0)


def random_distinct_indices(pop_size: int, target_idx: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices distinct from each other and from the target."""

    choices = np.delete(np.arange(pop_size), target_idx)
    if choices.size < count:
        raise ValueError(f"Need at least {count + 1} population members for this DE strategy.")
    return rng.choice(choices, size=count, replace=False)


def binomial_crossover(target: np.ndarray, mutant: np.ndarray, crossover_factor: float, rng: np.random.Generator) -> np.ndarray:
    """Apply binomial crossover and force at least one mutant dimension."""

    mask = rng.random(target.shape[0]) < crossover_factor
    mask[int(rng.integers(0, target.shape[0]))] = True
    return np.where(mask, mutant, target)


def exponential_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    crossover_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply exponential crossover and force at least one mutant dimension."""

    trial = target.copy()
    n = target.shape[0]
    start = int(rng.integers(0, n))
    length = 1
    while length < n and rng.random() < crossover_factor:
        length += 1
    for offset in range(length):
        trial[(start + offset) % n] = mutant[(start + offset) % n]
    return trial


def update_best(de: DEData, population: np.ndarray, chrom: np.ndarray, objv: np.ndarray) -> None:
    """Update global best state from the current accepted population."""

    best_idx = int(np.nanargmin(objv))
    best_val = float(objv[best_idx])
    if best_val <= de.fxmin:
        de.fxmin = best_val
        de.xmin = chrom[best_idx].copy()
        de.best_particle = population[best_idx].copy()


def print_generation(de: DEData) -> None:
    print("------------------------------------------------", flush=True)
    print(f"DE generation: {de.generation}", flush=True)
    print(f"   strategy: {STRATEGY_NAMES[de.strategy]}", flush=True)
    print(f"   xmin: {de.xmin.tolist() if de.xmin is not None else None} -- f(xmin): {de.fxmin}", flush=True)


def print_results(de: DEData) -> None:
    print("------------------------------------------------", flush=True)
    print("######   DE RESULT   #########", flush=True)
    print(f"   Objective function for xmin: {de.fxmin}", flush=True)
    print(f"   xmin: {de.xmin.tolist() if de.xmin is not None else None}", flush=True)
    print("------------------------------------------------", flush=True)


def save_generation_data(de: DEData) -> None:
    """Save DE history in `python_tempdata/tempdata.npz`."""

    out = optimizer_history_dir(de.config)
    de.history_population.append(de.population.copy())
    de.history_trial_population.append(de.trial_population.copy())
    de.history_chrom.append(de.chrom.copy())
    de.history_trial_chrom.append(de.trial_chrom.copy())
    de.history_obj.append(de.objv.copy())
    de.history_trial_obj.append(de.trial_objv.copy())
    payload = {
        "method": np.array("DE"),
        "DEstrategy": np.array(STRATEGY_NAMES[de.strategy]),
        "DEmutationFactor": np.array(float(de.mutation_factor), dtype=float),
        "DEcrossoverFactor": np.array(float(de.crossover_factor), dtype=float),
        "DEpop": np.array(de.history_population),
        "DEtrial": np.array(de.history_trial_population),
        "DEgen": np.array(de.history_chrom),
        "DEtrialgen": np.array(de.history_trial_chrom),
        "DEobj": np.array(de.history_obj),
        "DEtrialobj": np.array(de.history_trial_obj),
        "DEgenb": np.array(de.xmingen),
        "DEobjb": -np.array(de.fxmingen),
    }
    payload.update(
        common_metadata(
            "DE",
            de.config,
            de.generation,
            best_chromosome=de.xmin,
            best_objective=de.fxmin,
        )
    )
    payload.update(best_population_payload(de.chrom, de.objv, prefix="DE"))
    target = out / "tempdata.npz"
    atomic_savez(target, payload, fallback_stem=f"tempdata_generation_{de.generation:04d}")
