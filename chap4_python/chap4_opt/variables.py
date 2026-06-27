"""Decision-vector helpers for chapter 4.

The combined vector stores each well as `[active, type, i, j]`. The iterative
algorithm only optimizes `[active, type]` and then combines those bits with the
current reference locations from `locidx`.
"""

from __future__ import annotations

import numpy as np

from .config import CaseConfig


def inj_slice(cfg: CaseConfig, idx: int) -> slice:
    start = idx * (cfg.bpi + 2)
    return slice(start, start + cfg.bpi + 2)


def pro_slice(cfg: CaseConfig, idx: int) -> slice:
    start = cfg.num_inj * (cfg.bpi + 2) + idx * (cfg.bpp + 2)
    return slice(start, start + cfg.bpp + 2)


def well_slice(cfg: CaseConfig, idx: int) -> slice:
    if idx < cfg.num_inj:
        return inj_slice(cfg, idx)
    return pro_slice(cfg, idx - cfg.num_inj)


def form_var_ga(cfg: CaseConfig, chrom: np.ndarray) -> np.ndarray:
    """Python version of `form_var_GA.m` for iterative chromosomes."""

    if cfg.locidx is None:
        raise RuntimeError("form_var_ga needs locidx from baseinfo.mat.")
    chrom = np.asarray(chrom)
    if chrom.ndim == 1:
        chrom = chrom.reshape(1, -1)
    out = np.zeros((chrom.shape[0], cfg.combined_length), dtype=float)
    for r, row in enumerate(chrom):
        for widx in range(cfg.num_wells):
            dest = well_slice(cfg, widx)
            out[r, dest.start : dest.start + 2] = row[widx * 2 : widx * 2 + 2]
            out[r, dest.start + 2 : dest.stop] = cfg.locidx[widx, 1:3]
    return out


def is_oil_cell(cfg: CaseConfig, i_grid: int, j_grid: int) -> bool:
    """Return whether a 1-based grid cell is oil-bearing in `oilgb`."""

    if cfg.oilgb is None:
        return True
    i0 = int(i_grid) - 1
    j0 = int(j_grid) - 1
    if i0 < 0 or j0 < 0 or i0 >= cfg.oilgb.shape[0] or j0 >= cfg.oilgb.shape[1]:
        return False
    threshold = 1 if cfg.name == "Brugge" else 1
    return bool(cfg.oilgb[i0, j0] >= threshold)


def active_well_count(cfg: CaseConfig, u: np.ndarray) -> int:
    """Count active wells in a combined decision vector."""

    total = 0
    for widx in range(cfg.num_wells):
        total += int(round(u[well_slice(cfg, widx).start]))
    return total


def binary_indices(cfg: CaseConfig) -> np.ndarray:
    """Indices of `[active, type]` entries inside a combined chromosome."""

    indices: list[int] = []
    for widx in range(cfg.num_wells):
        sl = well_slice(cfg, widx)
        indices.extend([sl.start, sl.start + 1])
    return np.asarray(indices, dtype=int)


def repair_bounds(cfg: CaseConfig, chrom: np.ndarray) -> np.ndarray:
    """Clamp status/type to 0/1 and locations inside reservoir bounds."""

    fixed = np.asarray(chrom, dtype=float).copy()
    for widx in range(cfg.num_wells):
        sl = well_slice(cfg, widx)
        fixed[sl.start] = 1.0 if fixed[sl.start] >= 0.5 else 0.0
        fixed[sl.start + 1] = 1.0 if fixed[sl.start + 1] >= 0.5 else 0.0
        fixed[sl.start + 2] = float(np.clip(np.ceil(fixed[sl.start + 2]), 1, cfg.grid_i))
        fixed[sl.start + 3] = float(np.clip(np.ceil(fixed[sl.start + 3]), 1, cfg.grid_j))
    return fixed


def is_feasible(cfg: CaseConfig, chrom: np.ndarray) -> bool:
    """Check active grid cells and minimum inter-well distance."""

    if cfg.actnum is None:
        return True
    coords = []
    for widx in range(cfg.num_wells):
        sl = well_slice(cfg, widx)
        i_grid = int(chrom[sl.start + 2])
        j_grid = int(chrom[sl.start + 3])
        if i_grid < 1 or j_grid < 1 or i_grid > cfg.grid_i or j_grid > cfg.grid_j:
            return False
        if cfg.actnum[i_grid - 1, j_grid - 1] == 0:
            return False
        coords.append((i_grid, j_grid))
    coords_arr = np.asarray(coords, dtype=float)
    for i in range(len(coords_arr)):
        for j in range(i + 1, len(coords_arr)):
            if np.linalg.norm(coords_arr[i] - coords_arr[j]) < cfg.min_rad:
                return False
    return True


def nearest_active(cfg: CaseConfig, i_grid: int, j_grid: int, radius: int = 5) -> tuple[int, int]:
    """Map an invalid location to a nearby active grid cell."""

    if cfg.actnum is None:
        return int(np.clip(i_grid, 1, cfg.grid_i)), int(np.clip(j_grid, 1, cfg.grid_j))
    i_grid = int(np.clip(i_grid, 1, cfg.grid_i))
    j_grid = int(np.clip(j_grid, 1, cfg.grid_j))
    if cfg.actnum[i_grid - 1, j_grid - 1] != 0:
        return i_grid, j_grid
    best = (i_grid, j_grid)
    best_dist = float("inf")
    for ii in range(max(1, i_grid - radius), min(cfg.grid_i, i_grid + radius) + 1):
        for jj in range(max(1, j_grid - radius), min(cfg.grid_j, j_grid + radius) + 1):
            if cfg.actnum[ii - 1, jj - 1] == 0:
                continue
            dist = (ii - i_grid) ** 2 + (jj - j_grid) ** 2
            if dist < best_dist:
                best = (ii, jj)
                best_dist = dist
    return best


def repair_feasible(cfg: CaseConfig, chrom: np.ndarray) -> np.ndarray:
    """Heuristic repair used after mixed/real-valued crossover."""

    fixed = repair_bounds(cfg, chrom)
    for widx in range(cfg.num_wells):
        sl = well_slice(cfg, widx)
        i_grid, j_grid = nearest_active(cfg, int(fixed[sl.start + 2]), int(fixed[sl.start + 3]))
        fixed[sl.start + 2] = i_grid
        fixed[sl.start + 3] = j_grid
    return fixed
