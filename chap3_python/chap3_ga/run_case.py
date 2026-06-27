"""Programmatic runner used by case-level `setup_ga.py` files."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .case_setup import config_from_setup, load_setup_module
from .config import setup_report
from .encoding import decode_locations
from .ga import GAData, run_ga
from .initial_solutions import initial_chromosomes_from_setup
from .writers import is_forced_injector, selected_type
from .objective import ObjectiveEvaluator, prepare_work_folders


def run_from_setup(setup_file: str | Path) -> None:
    """Run a case directly from its `setup_ga.py` file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)
    prepare_work_folders(cfg)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        print(setup_report(cfg))
        return

    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_optimizer_job_info(cfg.work_dir, run_id, setup_file)
    print(
        f"Optimization job started: run_id={run_id}, python_pid={os.getpid()}, "
        f"case={cfg.name}, setup={setup_file}",
        flush=True,
    )

    objective = ObjectiveEvaluator(
        cfg,
        dry_run=bool(getattr(module, "DRY_RUN", False)),
        stream_simulator_output=bool(getattr(module, "STREAM_SIMULATOR_OUTPUT", False)),
        print_batch_timing=bool(getattr(module, "PRINT_BATCH_TIMING", True)),
        results_timeout_seconds=getattr(module, "RESULTS_TIMEOUT_SECONDS", 60.0),
        simulation_interrupt_timeout_seconds=getattr(
            module,
            "SIMULATION_INTERRUPT_TIMEOUT_SECONDS",
            60.0,
        ),
    )
    ga = GAData(cfg, objective, initial_chromosomes=initial_chromosomes_from_setup(module, cfg))
    run_ga(ga, seed=int(getattr(module, "SEED", 1000)))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)):
        update_baseinfo1_locidx(ga)


def write_optimizer_job_info(work_dir: Path, run_id: str, setup_file: Path) -> None:
    """Save the optimizer process identity for later checking."""

    info_dir = work_dir / "python_tempdata"
    info_dir.mkdir(parents=True, exist_ok=True)
    info = info_dir / "current_job.txt"
    info.write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"python_pid={os.getpid()}",
                f"setup={setup_file}",
            ]
        )
        + "\n"
    )


def update_baseinfo1_locidx(ga: GAData) -> None:
    """Write the best type/location result for a later order-only run."""

    cfg = ga.config
    if cfg.design_var != 2:
        print("Skipping baseinfo1_locidx.csv update because only DESIGN_VAR = 2 prepares that file.")
        return
    if cfg.locidx is None:
        raise RuntimeError("Cannot update baseinfo1_locidx.csv because locidx was not loaded.")
    if ga.xmin is None:
        raise RuntimeError("Cannot update baseinfo1_locidx.csv because the GA has no best chromosome.")

    locs = decode_locations(ga.xmin, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location)
    updated = np.full((cfg.locidx.shape[0], 4), -1, dtype=int)
    updated[:, : min(cfg.locidx.shape[1], 3)] = np.asarray(cfg.locidx[:, :3], dtype=int)

    for well_idx, loc in enumerate(locs[: cfg.num_wells]):
        row_idx = int(loc) - 1
        if row_idx < 0 or row_idx >= updated.shape[0]:
            raise RuntimeError(f"Optimized location {loc} is outside the baseinfo table.")
        inject = selected_type(cfg, ga.xmin, well_idx) or is_forced_injector(cfg, int(loc))
        updated[row_idx, 3] = 1 if inject else 0

    out = Path(cfg.work_dir) / "baseinfo1_locidx.csv"
    np.savetxt(out, updated, delimiter=",", fmt="%d")
    cfg.loaded_data_file = out
    print(f"Updated {out} from the best optimized type/location chromosome.")
