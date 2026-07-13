"""Particle Swarm Optimization for Chapter 3 design variables.

PSO moves continuous particles in normalized [0, 1] space. Each particle is
decoded into the same mixed-integer design representation used by the shared
objective pipeline before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from chap3_ga.config import CaseConfig
from chap3_ga.lhs_initialization import decode_lhs_population, lhs_population, normalized_dimension


Objective = Callable[[np.ndarray], np.ndarray]


@dataclass
class PSOData:
    """Mutable PSO state."""

    config: CaseConfig
    objective: Objective
    max_iterations: int | None = None
    swarm_size: int | None = None
    omega: float = 0.7298
    phip: float = 1.496
    phig: float = 1.496
    initial_particles: np.ndarray | None = None
    initial_velocity: str = "zero"
    particles: np.ndarray | None = None
    velocities: np.ndarray | None = None
    chrom: np.ndarray | None = None
    objv: np.ndarray | None = None
    personal_best_particles: np.ndarray | None = None
    personal_best_obj: np.ndarray | None = None
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    best_particle: np.ndarray | None = None
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    history_particles: list[np.ndarray] = field(default_factory=list)
    history_velocity: list[np.ndarray] = field(default_factory=list)
    history_chrom: list[np.ndarray] = field(default_factory=list)
    history_obj: list[np.ndarray] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self) -> None:
        self.max_iterations = self.config.maxgen if self.max_iterations is None else self.max_iterations
        self.swarm_size = self.config.population_size if self.swarm_size is None else self.swarm_size
        self.dimension = normalized_dimension(self.config)


def run_pso(pso: PSOData, seed: int = 1000, save_history: bool = True) -> PSOData:
    """Run PSO and return the final state."""

    rng = np.random.default_rng(seed)
    particles = initial_particles(pso, rng)
    velocities = initial_velocities(pso, rng)
    pbest_particles = particles.copy()
    pbest_obj = np.full(int(pso.swarm_size), np.inf, dtype=float)

    evaluate_swarm(pso, particles, velocities, pbest_particles, pbest_obj, iteration=0, save_history=save_history)

    for iteration in range(1, int(pso.max_iterations) + 1):
        rp = rng.uniform(size=particles.shape)
        rg = rng.uniform(size=particles.shape)
        if pso.best_particle is None:
            raise RuntimeError("Cannot update PSO velocity before a global best particle is available.")
        velocities = (
            float(pso.omega) * velocities
            + float(pso.phip) * rp * (pbest_particles - particles)
            + float(pso.phig) * rg * (pso.best_particle - particles)
        )
        particles = np.clip(particles + velocities, 0.0, 1.0)
        evaluate_swarm(
            pso,
            particles,
            velocities,
            pbest_particles,
            pbest_obj,
            iteration=iteration,
            save_history=save_history,
        )

    print_results(pso)
    return pso


def initial_particles(pso: PSOData, rng: np.random.Generator) -> np.ndarray:
    """Return the initial normalized swarm."""

    if pso.initial_particles is not None:
        particles = np.asarray(pso.initial_particles, dtype=float).copy()
    else:
        particles, _ = lhs_population(int(pso.swarm_size), pso.dimension, rng)
    expected = (int(pso.swarm_size), pso.dimension)
    if particles.shape != expected:
        raise ValueError(f"Initial PSO particles have shape {particles.shape}; expected {expected}.")
    return np.clip(particles, 0.0, 1.0)


def initial_velocities(pso: PSOData, rng: np.random.Generator) -> np.ndarray:
    """Return the initial normalized velocity matrix."""

    mode = str(pso.initial_velocity).strip().lower()
    shape = (int(pso.swarm_size), pso.dimension)
    if mode == "zero":
        return np.zeros(shape, dtype=float)
    if mode == "random":
        return rng.uniform(-1.0, 1.0, size=shape)
    raise ValueError(f"Unsupported INITIAL_VELOCITY={pso.initial_velocity!r}. Expected 'zero' or 'random'.")


def evaluate_swarm(
    pso: PSOData,
    particles: np.ndarray,
    velocities: np.ndarray,
    pbest_particles: np.ndarray,
    pbest_obj: np.ndarray,
    iteration: int,
    save_history: bool,
) -> None:
    """Decode/evaluate one swarm and update personal/global bests."""

    pso.iteration = iteration
    pso.particles = particles.copy()
    pso.velocities = velocities.copy()
    pso.chrom = decode_lhs_population(pso.config, particles)
    pso.objv = np.asarray(pso.objective(pso.chrom), dtype=float)

    improved = pso.objv < pbest_obj
    pbest_particles[improved] = particles[improved]
    pbest_obj[improved] = pso.objv[improved]
    pso.personal_best_particles = pbest_particles.copy()
    pso.personal_best_obj = pbest_obj.copy()

    best_idx = int(np.argmin(pbest_obj))
    best_val = float(pbest_obj[best_idx])
    if best_val <= pso.fxmin:
        pso.best_particle = pbest_particles[best_idx].copy()
        pso.fxmin = best_val
        pso.xmin = decode_lhs_population(pso.config, pso.best_particle.reshape(1, -1))[0]

    pso.xmingen.append(pso.xmin.copy())
    pso.fxmingen.append(float(pso.fxmin))
    if save_history:
        save_iteration_data(pso)
    print_iteration(pso)


def print_iteration(pso: PSOData) -> None:
    """Print PSO progress in the same spirit as GA/ILHS."""

    print(f"Iteration: {pso.iteration}", flush=True)
    print(f"   xmin: {pso.xmin.tolist() if pso.xmin is not None else None} -- f(xmin): {pso.fxmin}", flush=True)


def print_results(pso: PSOData) -> None:
    """Print final PSO result."""

    print("PSO optimization completed", flush=True)
    print(f"   Objective function for xmin: {pso.fxmin}", flush=True)
    print(f"   xmin: {pso.xmin.tolist() if pso.xmin is not None else None}", flush=True)


def save_iteration_data(pso: PSOData) -> None:
    """Save PSO history to the case work directory."""

    out = Path(pso.config.work_dir) / "python_tempdata"
    out.mkdir(parents=True, exist_ok=True)
    pso.history_particles.append(pso.particles.copy())
    pso.history_velocity.append(pso.velocities.copy())
    pso.history_chrom.append(pso.chrom.copy())
    pso.history_obj.append(pso.objv.copy())
    data = {
        "method": np.array("pso"),
        "PSOparticles": np.array(pso.history_particles),
        "PSOvelocity": np.array(pso.history_velocity),
        "PSOgen": np.array(pso.history_chrom),
        "PSOgenb": np.array(pso.xmingen),
        "PSOobj": np.array(pso.history_obj),
        "PSOobjb": -np.array(pso.fxmingen),
        "PSOpbest": np.array(pso.personal_best_particles),
        "PSOpbestobj": np.array(pso.personal_best_obj),
    }
    target = out / "tempdata.npz"
    try:
        np.savez(target, **data)
    except OSError as exc:
        fallback = out / f"tempdata_iteration_{pso.iteration:04d}.npz"
        np.savez(fallback, **data)
        print(f"Warning: could not update {target}: {exc}. Saved {fallback} instead.", flush=True)
