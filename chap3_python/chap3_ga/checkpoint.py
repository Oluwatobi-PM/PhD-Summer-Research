"""Checkpoint helpers shared by Chapter 3 optimizers."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .config import CaseConfig


CHECKPOINT_VERSION = 2


def optimizer_history_dir(cfg: CaseConfig) -> Path:
    """Return the directory used for optimizer history files."""

    out = Path(cfg.work_dir) / "python_tempdata"
    out.mkdir(parents=True, exist_ok=True)
    return out


def atomic_savez(
    target: Path,
    payload: dict[str, object],
    *,
    compressed: bool = True,
    fallback_stem: str = "tempdata",
    attempts: int = 5,
) -> None:
    """Write a NumPy checkpoint atomically, with a fallback for locked files."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    save = np.savez_compressed if compressed else np.savez
    save(tmp, **payload)
    for attempt in range(attempts):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                fallback = target.with_name(f"{fallback_stem}_{int(time.time())}{target.suffix}")
                tmp.replace(fallback)
                print(
                    f"Warning: could not update locked checkpoint {target}. "
                    f"Saved the latest checkpoint to {fallback} instead.",
                    flush=True,
                )
                return
            time.sleep(0.5)


def common_metadata(
    method: str,
    cfg: CaseConfig,
    iteration: int,
    *,
    best_chromosome: np.ndarray | None,
    best_objective: float,
) -> dict[str, object]:
    """Return scalar metadata that makes checkpoints self-describing."""

    payload: dict[str, object] = {
        "checkpoint_version": np.array(CHECKPOINT_VERSION, dtype=int),
        "method": np.array(method),
        "optimizer": np.array(method.lower()),
        "iteration": np.array(int(iteration), dtype=int),
        "case_name": np.array(str(cfg.name)),
        "design_var": np.array(int(cfg.design_var), dtype=int),
        "num_wells": np.array(int(cfg.num_wells), dtype=int),
        "num_locations": np.array(int(cfg.num_locations), dtype=int),
        "chromosome_length": np.array(int(cfg.chromosome_length), dtype=int),
        "objective_scaling": np.array(float(cfg.objective_scaling), dtype=float),
        "objective_sign": np.array("minimize_negative_npv"),
        "best_objective": np.array(float(best_objective), dtype=float),
        "best_npv_scaled": np.array(-float(best_objective), dtype=float),
    }
    if best_chromosome is not None:
        payload["best_chromosome"] = np.asarray(best_chromosome, dtype=int).copy()
    return payload


def best_population_payload(
    chrom: np.ndarray,
    objv: np.ndarray,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return UOF-like best-population arrays for optimizer handoff/restarts."""

    chrom_arr = np.asarray(chrom, dtype=int).copy()
    obj_arr = np.asarray(objv, dtype=float).reshape(-1).copy()
    order = np.argsort(obj_arr)
    return {
        "best_population_chrom": chrom_arr[order],
        "best_population_obj": obj_arr[order],
        f"{prefix}bestPopulation": chrom_arr[order],
        f"{prefix}bestPopulationObj": obj_arr[order],
    }


def require_checkpoint_arrays(data: np.lib.npyio.NpzFile, required: set[str]) -> None:
    """Raise a readable error if a checkpoint is missing required arrays."""

    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"Restart file is missing required arrays: {', '.join(missing)}")
