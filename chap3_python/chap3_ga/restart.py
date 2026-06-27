"""Restart helpers for continuing a saved GA optimization run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ga import GAData, run_ga


def load_restart_state(ga: GAData, restart_file: str | Path) -> int:
    """Load saved GA history and use the last saved population as restart state.

    Returns the number of generations already present in the checkpoint.
    """

    restart_path = Path(restart_file).resolve()
    data = np.load(restart_path, allow_pickle=True)
    required = {"GAgen", "GAobj", "GAgenb", "GAobjb"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"Restart file is missing required arrays: {', '.join(missing)}")

    history_chrom = np.asarray(data["GAgen"], dtype=int)
    history_obj = np.asarray(data["GAobj"], dtype=float)
    history_best_chrom = np.asarray(data["GAgenb"], dtype=int)
    history_best_npv = np.asarray(data["GAobjb"], dtype=float)

    if history_chrom.ndim != 3:
        raise ValueError("GAgen must have shape generations x population x chromosome_length.")
    if history_chrom.shape[0] == 0:
        raise ValueError("Restart file has no saved generations.")
    if history_chrom.shape[1] != ga.nind:
        raise ValueError(f"Restart population size {history_chrom.shape[1]} does not match configured {ga.nind}.")
    if history_chrom.shape[2] != ga.config.chromosome_length:
        raise ValueError(
            f"Restart chromosome length {history_chrom.shape[2]} does not match configured "
            f"{ga.config.chromosome_length}."
        )

    ga.history_chrom = [row.copy() for row in history_chrom]
    ga.history_obj = [row.copy() for row in history_obj]
    ga.xmingen = [row.copy() for row in history_best_chrom]
    ga.fxmingen = [-float(value) for value in history_best_npv]

    best_idx = int(np.argmin(ga.fxmingen))
    ga.fxmin = float(ga.fxmingen[best_idx])
    ga.xmin = np.asarray(ga.xmingen[best_idx], dtype=int).copy()
    ga.chrom = history_chrom[-1].copy()

    return int(history_chrom.shape[0])


def run_ga_restart(ga: GAData, restart_file: str | Path, extra_generations: int, seed: int = 1000) -> GAData:
    """Continue a GA run from a saved `tempdata.npz` checkpoint."""

    completed = load_restart_state(ga, restart_file)
    ga.maxgen = int(extra_generations)
    print(
        f"Restarting GA from {restart_file}: {completed} saved generation(s), "
        f"running {extra_generations} additional generation(s).",
        flush=True,
    )
    return run_ga(ga, seed=seed, generation_offset=completed)
