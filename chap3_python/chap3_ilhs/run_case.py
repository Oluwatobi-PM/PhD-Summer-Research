"""Programmatic runner used by case-level `setup_ilhs.py` files."""

from __future__ import annotations

import os
import time
from pathlib import Path

from chap3_ga.case_setup import load_setup_module
from chap3_ga.config import setup_report
from chap3_ga.objective import ObjectiveEvaluator, clean_generated_work_folders, prepare_work_folders
from chap3_ga.run_case import update_baseinfo1_locidx, write_optimizer_job_info

from .case_setup import config_from_setup
from .ilhs import ILHSData, normalized_dimension, run_ilhs


def run_from_setup(setup_file: str | Path) -> None:
    """Run an ILHS case directly from its `setup_ilhs.py` file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        print(setup_report(cfg))
        print(f"ilhs_dimensions: {normalized_dimension(cfg)}")
        print(f"number_of_samples: {cfg.population_size}")
        print(f"max_iterations: {cfg.maxgen}")
        return

    if bool(getattr(module, "CLEAN_WORK_FOLDERS_ON_START", True)):
        clean_generated_work_folders(
            cfg,
            clean_history=bool(getattr(module, "CLEAN_HISTORY_ON_START", True)),
        )
    prepare_work_folders(cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_optimizer_job_info(cfg.work_dir, run_id, setup_file)
    print(
        f"ILHS optimization job started: run_id={run_id}, python_pid={os.getpid()}, "
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
    ilhs = ILHSData(
        cfg,
        objective,
        max_iterations=cfg.maxgen,
        number_of_samples=cfg.population_size,
        entropy=float(getattr(module, "ENTROPY", 0.9)),
    )
    run_ilhs(ilhs, seed=int(getattr(module, "SEED", 1000)))

    dry_run = bool(getattr(module, "DRY_RUN", False))
    allow_dry_update = bool(getattr(module, "ALLOW_DRY_RUN_BASEINFO1_UPDATE", False))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)) and (not dry_run or allow_dry_update):
        update_baseinfo1_locidx(ilhs)
    elif dry_run and not allow_dry_update:
        print("Skipping baseinfo1_locidx.csv update because this was a dry run.", flush=True)
