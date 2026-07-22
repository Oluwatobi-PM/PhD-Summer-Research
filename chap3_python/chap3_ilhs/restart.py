"""Restart helpers for continuing saved ILHS optimization runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chap3_ga.checkpoint import require_checkpoint_arrays

from .ilhs import ILHSData, next_jointopt_population, run_ilhs


def load_restart_state(ilhs: ILHSData, restart_file: str | Path, seed: int = 1000) -> int:
    """Load an ILHS checkpoint and prepare the next sampling population."""

    restart_path = Path(restart_file).resolve()
    data = np.load(restart_path, allow_pickle=False)
    require_checkpoint_arrays(data, {"ILHSgen", "ILHSorder", "ILHSbounds", "GAgen", "GAobj", "GAgenb", "GAobjb"})

    particles = np.asarray(data["ILHSgen"], dtype=float)
    order = np.asarray(data["ILHSorder"], dtype=int)
    bounds = np.asarray(data["ILHSbounds"], dtype=float)
    chrom = np.asarray(data["GAgen"], dtype=int)
    obj = np.asarray(data["GAobj"], dtype=float)
    best_chrom = np.asarray(data["GAgenb"], dtype=int)
    best_npv = np.asarray(data["GAobjb"], dtype=float).reshape(-1)

    if particles.ndim != 3:
        raise ValueError("ILHSgen must have shape iterations x samples x dimensions.")
    expected = (int(ilhs.number_of_samples), ilhs.dimension)
    if particles.shape[1:] != expected:
        raise ValueError(f"Restart ILHS shape {particles.shape[1:]} does not match configured {expected}.")
    if order.shape != particles.shape:
        raise ValueError("ILHSorder must have the same shape as ILHSgen.")
    if bounds.shape != particles.shape:
        raise ValueError("ILHSbounds must have the same shape as ILHSgen.")
    if chrom.shape[:2] != particles.shape[:2]:
        raise ValueError("GAgen sample history does not align with ILHSgen.")
    if obj.shape[:2] != particles.shape[:2]:
        raise ValueError("GAobj objective history does not align with ILHSgen.")

    ilhs.history_particles = [row.copy() for row in particles]
    ilhs.history_order = [row.copy() for row in order]
    ilhs.history_bounds = [row.copy() for row in bounds]
    ilhs.history_chrom = [row.copy() for row in chrom]
    ilhs.history_obj = [row.copy() for row in obj]
    ilhs.xmingen = [row.copy() for row in best_chrom]
    ilhs.fxmingen = [-float(value) for value in best_npv]
    ilhs.xmin = ilhs.xmingen[-1].copy()
    ilhs.fxmin = ilhs.fxmingen[-1]

    rng = np.random.default_rng(seed)
    next_particles, next_order, next_bounds = next_jointopt_population(
        particles[-1],
        order[-1],
        bounds[-1],
        obj[-1],
        int(ilhs.number_of_samples),
        ilhs.dimension,
        ilhs.entropy,
        rng,
    )
    ilhs.initial_particles = next_particles
    ilhs.initial_order = next_order
    ilhs.initial_bounds = next_bounds
    return int(particles.shape[0])


def run_ilhs_restart(ilhs: ILHSData, restart_file: str | Path, extra_iterations: int, seed: int = 1000) -> ILHSData:
    """Continue an ILHS run from a saved `tempdata.npz` checkpoint."""

    completed = load_restart_state(ilhs, restart_file, seed=seed)
    ilhs.max_iterations = int(extra_iterations)
    print(
        f"Restarting ILHS from {restart_file}: {completed} saved iteration(s), "
        f"running {extra_iterations} additional iteration(s).",
        flush=True,
    )
    return run_ilhs(ilhs, seed=seed, iteration_offset=completed)
