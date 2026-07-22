"""Restart helpers for continuing saved PSO optimization runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chap3_ga.checkpoint import require_checkpoint_arrays
from chap3_ga.lhs_initialization import decode_lhs_population

from .pso import PSOData, run_pso


def load_restart_state(pso: PSOData, restart_file: str | Path) -> int:
    """Load a PSO checkpoint and return the number of saved iterations."""

    restart_path = Path(restart_file).resolve()
    data = np.load(restart_path, allow_pickle=False)
    require_checkpoint_arrays(data, {"PSOparticles", "PSOvelocity", "PSOpbest", "PSOpbestobj", "PSOobjb"})

    particles = np.asarray(data["PSOparticles"], dtype=float)
    velocities = np.asarray(data["PSOvelocity"], dtype=float)
    pbest_particles = np.asarray(data["PSOpbest"], dtype=float)
    pbest_obj = np.asarray(data["PSOpbestobj"], dtype=float).reshape(-1)
    best_npv = np.asarray(data["PSOobjb"], dtype=float).reshape(-1)

    if particles.ndim != 3:
        raise ValueError("PSOparticles must have shape iterations x swarm_size x dimensions.")
    expected = (int(pso.swarm_size), pso.dimension)
    if particles.shape[1:] != expected:
        raise ValueError(f"Restart swarm shape {particles.shape[1:]} does not match configured {expected}.")
    if velocities.shape != particles.shape:
        raise ValueError("PSOvelocity must have the same shape as PSOparticles.")
    if pbest_particles.shape != expected:
        raise ValueError(f"PSOpbest shape {pbest_particles.shape} does not match configured {expected}.")
    if pbest_obj.shape[0] != expected[0]:
        raise ValueError("PSOpbestobj length must match swarm size.")

    pso.history_particles = [row.copy() for row in particles]
    pso.history_velocity = [row.copy() for row in velocities]
    if "PSOgen" in data:
        pso.history_chrom = [row.copy() for row in np.asarray(data["PSOgen"], dtype=int)]
    if "PSOobj" in data:
        pso.history_obj = [row.copy() for row in np.asarray(data["PSOobj"], dtype=float)]

    pso.particles = particles[-1].copy()
    pso.velocities = velocities[-1].copy()
    pso.personal_best_particles = pbest_particles.copy()
    pso.personal_best_obj = pbest_obj.copy()
    pso.fxmingen = [-float(value) for value in best_npv]

    if "PSOgenb" in data:
        pso.xmingen = [row.copy() for row in np.asarray(data["PSOgenb"], dtype=int)]
        pso.xmin = pso.xmingen[-1].copy()
    elif "best_chromosome" in data:
        pso.xmin = np.asarray(data["best_chromosome"], dtype=int).copy()
        pso.xmingen = [pso.xmin.copy()]
    else:
        best_idx = int(np.nanargmin(pbest_obj))
        pso.xmin = decode_lhs_population(pso.config, pbest_particles[best_idx].reshape(1, -1))[0]
        pso.xmingen = [pso.xmin.copy()]

    pso.fxmin = -float(best_npv[-1])
    best_idx = int(np.nanargmin(pbest_obj))
    pso.best_particle = pbest_particles[best_idx].copy()
    return int(particles.shape[0])


def run_pso_restart(pso: PSOData, restart_file: str | Path, extra_iterations: int, seed: int = 1000) -> PSOData:
    """Continue a PSO run from a saved `tempdata.npz` checkpoint."""

    completed = load_restart_state(pso, restart_file)
    pso.max_iterations = int(extra_iterations)
    print(
        f"Restarting PSO from {restart_file}: {completed} saved iteration(s), "
        f"running {extra_iterations} additional iteration(s).",
        flush=True,
    )
    return run_pso(pso, seed=seed, iteration_offset=completed - 1, evaluate_initial=False)
