"""Iterative Latin Hypercube Sampling for Chapter 3 design variables.

ILHS samples normalized values in [0, 1], decodes them into the same
mixed Brugge/channelmodel chromosome used by the GA, and evaluates that
chromosome with the existing objective pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import math
import time

import numpy as np

from chap3_ga.config import CaseConfig
from chap3_ga.lhs_initialization import (
    decode_lhs_particle,
    decode_lhs_population,
    decode_unique_locations,
    lhs_population,
    normalized_dimension,
    rank_order,
)


Objective = Callable[[np.ndarray], np.ndarray]


@dataclass
class ILHSData:
    """Mutable ILHS state."""

    config: CaseConfig
    objective: Objective
    max_iterations: int | None = None
    number_of_samples: int | None = None
    entropy: float = 0.9
    particles: np.ndarray | None = None
    initial_particles: np.ndarray | None = None
    initial_order: np.ndarray | None = None
    chrom: np.ndarray | None = None
    objv: np.ndarray | None = None
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    history_chrom: list[np.ndarray] = field(default_factory=list)
    history_particles: list[np.ndarray] = field(default_factory=list)
    history_order: list[np.ndarray] = field(default_factory=list)
    history_bounds: list[np.ndarray] = field(default_factory=list)
    history_obj: list[np.ndarray] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self) -> None:
        self.max_iterations = self.config.maxgen if self.max_iterations is None else self.max_iterations
        self.number_of_samples = (
            self.config.population_size if self.number_of_samples is None else self.number_of_samples
        )
        self.dimension = normalized_dimension(self.config)


def run_ilhs(ilhs: ILHSData, seed: int = 1000, save_history: bool = True) -> ILHSData:
    """Run the full ILHS loop and return the final state."""

    rng = np.random.default_rng(seed)
    if ilhs.initial_particles is None:
        particles, order = initial_population(int(ilhs.number_of_samples), ilhs.dimension, rng)
    else:
        particles = np.asarray(ilhs.initial_particles, dtype=float).copy()
        order = (
            np.asarray(ilhs.initial_order, dtype=int).copy()
            if ilhs.initial_order is not None
            else np.zeros_like(particles, dtype=int)
        )
    old_bounds = np.array(
        [
            [i / int(ilhs.number_of_samples) for _ in range(ilhs.dimension)]
            for i in range(1, int(ilhs.number_of_samples) + 1)
        ],
        dtype=float,
    )

    for iteration in range(int(ilhs.max_iterations)):
        ilhs.iteration = iteration
        ilhs.particles = particles.copy()
        ilhs.chrom = decode_population(ilhs.config, ilhs.particles, rng)
        ilhs.objv = ilhs.objective(ilhs.chrom)

        best_idx = int(np.argmin(ilhs.objv))
        best_val = float(ilhs.objv[best_idx])
        if best_val <= ilhs.fxmin:
            ilhs.xmin = ilhs.chrom[best_idx].copy()
            ilhs.fxmin = best_val

        ilhs.xmingen.append(ilhs.xmin.copy())
        ilhs.fxmingen.append(float(ilhs.fxmin))
        if save_history:
            save_iteration_data(ilhs, order, old_bounds)
        print_iteration(ilhs)

        particles, order, old_bounds = next_jointopt_population(
            particles,
            order,
            old_bounds,
            ilhs.objv,
            int(ilhs.number_of_samples),
            ilhs.dimension,
            ilhs.entropy,
            rng,
        )

    print_results(ilhs)
    return ilhs


def normalized_dimension(cfg: CaseConfig) -> int:
    """Return the number of normalized ILHS variables for the active design."""

    if cfg.design_var == 1:
        return 3 * cfg.num_wells
    if cfg.design_var == 2:
        return 2 * cfg.num_wells
    if cfg.design_var == 3:
        return cfg.num_wells
    raise ValueError(f"Unsupported design_var: {cfg.design_var}")


def initial_population(
    number_of_samples: int, number_of_dimensions: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """JointOpt `Initial_population`: LHS samples plus grid order record."""

    return lhs_population(number_of_samples, number_of_dimensions, rng)


def decode_population(cfg: CaseConfig, particles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decode normalized ILHS particles into GA-compatible chromosomes."""

    return decode_lhs_population(cfg, particles)


def decode_particle(cfg: CaseConfig, particle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decode one normalized particle into one mixed chromosome."""

    return decode_lhs_particle(cfg, particle)


def next_jointopt_population(
    particles: np.ndarray,
    order: np.ndarray,
    old_bounds: np.ndarray,
    objv: np.ndarray,
    number_of_samples: int,
    number_of_dimensions: int,
    entropy: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JointOpt ILHS update: rank weights, CDF points/bounds, shuffle order."""

    rank_index = rank(objv)
    gamma = compute_gamma(entropy, number_of_samples, 1.0)

    ranked_gamma = np.zeros(number_of_samples, dtype=float)
    for i in range(1, number_of_samples + 1):
        ranked_gamma[int(rank_index[i - 1]) - 1] = i ** (-gamma)
    normalizer = float(np.sum(ranked_gamma))

    ranked_order = np.zeros((number_of_samples, number_of_dimensions), dtype=float)
    for j in range(1, number_of_dimensions + 1):
        for i in range(1, number_of_samples + 1):
            ranked_order[int(order[i - 1, j - 1]) - 1, j - 1] = ranked_gamma[i - 1]

    new_points = np.zeros((number_of_samples, number_of_dimensions), dtype=float)
    new_bounds = np.zeros((number_of_samples, number_of_dimensions), dtype=float)
    for n in range(number_of_dimensions):
        points, bounds = cdf(old_bounds[:, n], ranked_order[:, n], normalizer, number_of_samples, rng)
        new_points[:, n] = points
        new_bounds[:, n] = bounds

    next_order = np.zeros((number_of_samples, number_of_dimensions), dtype=int)
    for j in range(number_of_dimensions):
        next_order[:, j] = rng.permutation(number_of_samples) + 1

    next_particles = np.zeros((number_of_samples, number_of_dimensions), dtype=float)
    for j in range(1, number_of_dimensions + 1):
        for k in range(1, number_of_samples + 1):
            next_particles[k - 1, j - 1] = new_points[int(next_order[k - 1, j - 1]) - 1, j - 1]

    return next_particles, next_order, new_bounds


def rank(values: np.ndarray) -> np.ndarray:
    """JointOpt `Rank`: sorted objective indices plus one."""

    return np.asarray(values).argsort() + 1


def compute_gamma(entropy: float, number_of_samples: int, y0: float) -> float:
    """JointOpt `Compute_gamma`, selecting the positive Zipf exponent.

    The entropy equation has both positive and negative roots for common
    ILHS entropy values. Only the positive root is consistent with
    w_i = r_i ** (-gamma), where rank 1 is the best sample and must receive
    the largest weight.
    """

    return solve_gamma_bisection(entropy, number_of_samples)


def gamma_equation(y: float, entropy: float, number_of_samples: int) -> float:
    return entropy - normalized_zipf_entropy(number_of_samples, y)


def solve_gamma_bisection(entropy: float, number_of_samples: int) -> float:
    """Solve for the positive gamma root with adaptive bracketing."""

    if number_of_samples <= 1:
        raise ValueError("number_of_samples must be greater than 1")
    if not 0.0 < entropy < 1.0:
        raise ValueError("entropy must be between 0 and 1")

    lo = 0.0
    hi = 1.0
    while normalized_zipf_entropy(number_of_samples, hi) > entropy:
        hi *= 2.0
        if hi > 1.0e6:
            raise RuntimeError("Could not bracket positive ILHS gamma root")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        h_mid = normalized_zipf_entropy(number_of_samples, mid)
        if abs(h_mid - entropy) < 1.0e-12:
            return mid
        if h_mid > entropy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normalized_zipf_entropy(number_of_samples: int, gamma: float) -> float:
    """Return the normalized Zipf entropy h(Ns, gamma)."""

    z = sum(float(i) ** (-gamma) for i in range(1, number_of_samples + 1))
    sm = sum(math.log(i) / (float(i) ** gamma) for i in range(1, number_of_samples + 1))
    return (1.0 / math.log(number_of_samples)) * ((gamma / z) * sm + math.log(z))


def cdf(
    prev_bounds: np.ndarray,
    rank_ord: np.ndarray,
    normalizer: float,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """JointOpt `CDF`: create new ordered points and bounds from rank weights."""

    points = np.zeros(num_samples, dtype=float)
    bounds = np.zeros(num_samples, dtype=float)
    cum_weights = np.zeros(num_samples + 1, dtype=float)

    for i in range(1, num_samples + 1):
        cum_weights[i] = cum_weights[i - 1] + rank_ord[i - 1] / normalizer

    prev_bounds = np.insert(prev_bounds, 0, 0.0)

    k = 1
    for i in range(1, num_samples + 1):
        alpha = (i - rng.uniform(0.0, 1.0)) / num_samples
        for j in range(k, num_samples + 1):
            if alpha >= cum_weights[j - 1] and (alpha < cum_weights[j] or j == num_samples):
                points[i - 1] = prev_bounds[j - 1] + (
                    (alpha - cum_weights[j - 1])
                    / (cum_weights[j] - cum_weights[j - 1])
                    * (prev_bounds[j] - prev_bounds[j - 1])
                )
                k = j

    k = 1
    for i in range(1, num_samples):
        alpha = float(i) / float(num_samples)
        for j in range(k, num_samples + 1):
            if alpha >= cum_weights[j - 1] and (alpha < cum_weights[j] or j == num_samples):
                bounds[i - 1] = prev_bounds[j - 1] + (
                    (alpha - cum_weights[j - 1])
                    / (cum_weights[j] - cum_weights[j - 1])
                    * (prev_bounds[j] - prev_bounds[j - 1])
                )
                k = j

    bounds[num_samples - 1] = 1.0
    return points, bounds


def print_iteration(ilhs: ILHSData) -> None:
    print("------------------------------------------------")
    print(f"ILHS iteration: {ilhs.iteration}")
    print(f"   xmin: {ilhs.xmin.tolist()} -- f(xmin): {ilhs.fxmin}")


def print_results(ilhs: ILHSData) -> None:
    print("------------------------------------------------")
    print("######   ILHS RESULT   #########")
    print(f"   Objective function for xmin: {ilhs.fxmin}")
    print(f"   xmin: {ilhs.xmin.tolist() if ilhs.xmin is not None else None}")
    print("------------------------------------------------")


def save_iteration_data(ilhs: ILHSData, order: np.ndarray, old_bounds: np.ndarray) -> None:
    """Save MATLAB-like history in `python_tempdata/tempdata.npz`."""

    out = Path(ilhs.config.work_dir) / "python_tempdata"
    out.mkdir(parents=True, exist_ok=True)
    ilhs.history_particles.append(ilhs.particles.copy())
    ilhs.history_order.append(order.copy())
    ilhs.history_bounds.append(old_bounds.copy())
    ilhs.history_chrom.append(ilhs.chrom.copy())
    ilhs.history_obj.append(ilhs.objv.copy())
    payload = {
        "method": np.array("ILHS"),
        "ILHSgen": np.array(ilhs.history_particles),
        "ILHSorder": np.array(ilhs.history_order),
        "ILHSbounds": np.array(ilhs.history_bounds),
        "GAgen": np.array(ilhs.history_chrom),
        "GAgenb": np.array(ilhs.xmingen),
        "GAobj": np.array(ilhs.history_obj),
        "GAobjb": -np.array(ilhs.fxmingen),
    }
    target = out / "tempdata.npz"
    tmp = out / "tempdata.tmp.npz"
    np.savez_compressed(tmp, **payload)
    for attempt in range(5):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == 4:
                print(
                    f"Warning: could not update locked checkpoint {target}. "
                    f"The latest checkpoint remains in {tmp}."
                )
                return
            time.sleep(0.5)
