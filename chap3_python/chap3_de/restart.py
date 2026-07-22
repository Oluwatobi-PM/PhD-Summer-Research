"""Restart helpers for continuing saved DE optimization runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chap3_ga.checkpoint import require_checkpoint_arrays

from .de import DEData, run_de


def load_restart_state(de: DEData, restart_file: str | Path) -> int:
    """Load a DE checkpoint and return the number of saved generations."""

    restart_path = Path(restart_file).resolve()
    data = np.load(restart_path, allow_pickle=False)
    require_checkpoint_arrays(data, {"DEpop", "DEgen", "DEobj", "DEgenb", "DEobjb"})

    population = np.asarray(data["DEpop"], dtype=float)
    chrom = np.asarray(data["DEgen"], dtype=int)
    obj = np.asarray(data["DEobj"], dtype=float)
    best_chrom = np.asarray(data["DEgenb"], dtype=int)
    best_npv = np.asarray(data["DEobjb"], dtype=float).reshape(-1)

    if population.ndim != 3:
        raise ValueError("DEpop must have shape generations x population_size x dimensions.")
    expected = (int(de.population_size), de.dimension)
    if population.shape[1:] != expected:
        raise ValueError(f"Restart DE population shape {population.shape[1:]} does not match configured {expected}.")
    if chrom.shape[:2] != population.shape[:2]:
        raise ValueError("DEgen chromosome history does not align with DEpop.")
    if obj.shape[:2] != population.shape[:2]:
        raise ValueError("DEobj objective history does not align with DEpop.")

    de.history_population = [row.copy() for row in population]
    if "DEtrial" in data:
        de.history_trial_population = [row.copy() for row in np.asarray(data["DEtrial"], dtype=float)]
    de.history_chrom = [row.copy() for row in chrom]
    if "DEtrialgen" in data:
        de.history_trial_chrom = [row.copy() for row in np.asarray(data["DEtrialgen"], dtype=int)]
    de.history_obj = [row.copy() for row in obj]
    if "DEtrialobj" in data:
        de.history_trial_obj = [row.copy() for row in np.asarray(data["DEtrialobj"], dtype=float)]
    de.xmingen = [row.copy() for row in best_chrom]
    de.fxmingen = [-float(value) for value in best_npv]

    de.population = population[-1].copy()
    de.chrom = chrom[-1].copy()
    de.objv = obj[-1].copy()
    de.xmin = de.xmingen[-1].copy()
    de.fxmin = de.fxmingen[-1]
    best_idx = int(np.nanargmin(de.objv))
    de.best_particle = de.population[best_idx].copy()
    return int(population.shape[0])


def run_de_restart(de: DEData, restart_file: str | Path, extra_generations: int, seed: int = 1000) -> DEData:
    """Continue a DE run from a saved `tempdata.npz` checkpoint."""

    completed = load_restart_state(de, restart_file)
    de.max_generations = int(extra_generations)
    print(
        f"Restarting DE from {restart_file}: {completed} saved generation(s), "
        f"running {extra_generations} additional generation(s).",
        flush=True,
    )
    return run_de(de, seed=seed, generation_offset=completed - 1, evaluate_initial=False)
