"""Programmatic runner used by case-level DE setup files."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from chap3_ga.case_setup import load_setup_module
from chap3_ga.config import setup_report
from chap3_ga.lhs_initialization import lhs_population, normalized_dimension
from chap3_ga.objective import ObjectiveEvaluator, clean_generated_work_folders, prepare_work_folders
from chap3_ga.run_case import update_baseinfo1_locidx, write_optimizer_job_info

from .case_setup import config_from_setup
from .de import DEData, STRATEGY_NAMES, run_de
from .restart import run_de_restart


def run_from_setup(setup_file: str | Path) -> None:
    """Run a DE case directly from its setup file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        print(setup_report(cfg))
        print(f"de_dimensions: {normalized_dimension(cfg)}")
        print(f"population_size: {cfg.population_size}")
        print(f"max_generations: {cfg.maxgen}")
        print(f"de_strategy: {STRATEGY_NAMES[int(getattr(module, 'DE_STRATEGY', 7))]}")
        return

    if not getattr(module, "RESTART_FROM", None) and bool(getattr(module, "CLEAN_WORK_FOLDERS_ON_START", True)):
        clean_generated_work_folders(
            cfg,
            clean_history=bool(getattr(module, "CLEAN_HISTORY_ON_START", True)),
        )
    prepare_work_folders(cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_optimizer_job_info(cfg.work_dir, run_id, setup_file, module, cfg)
    print(
        f"DE optimization job started: run_id={run_id}, python_pid={os.getpid()}, "
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

    initial = None
    initialization = str(getattr(module, "INITIALIZATION", "lhs")).strip().lower()
    if initialization == "lhs":
        init_seed = int(getattr(module, "INITIALIZATION_SEED", getattr(module, "SEED", 1000)))
        init_rng = np.random.default_rng(init_seed)
        initial, _ = lhs_population(cfg.population_size, normalized_dimension(cfg), init_rng)
    elif initialization == "random":
        initial = None
    else:
        raise ValueError(f"Unsupported INITIALIZATION={initialization!r}. Expected 'lhs' or 'random'.")

    de = DEData(
        cfg,
        objective,
        max_generations=cfg.maxgen,
        population_size=cfg.population_size,
        mutation_factor=float(getattr(module, "DE_MUTATION_FACTOR", getattr(module, "MUTATION_FACTOR", 0.7))),
        crossover_factor=float(getattr(module, "DE_CROSSOVER_FACTOR", getattr(module, "CROSSOVER_FACTOR", 0.5))),
        strategy=int(getattr(module, "DE_STRATEGY", 7)),
        initial_population=initial,
    )
    restart_from = getattr(module, "RESTART_FROM", None)
    if restart_from:
        extra_generations = getattr(module, "EXTRA_GENERATIONS", getattr(module, "EXTRA_ITERATIONS", None))
        if extra_generations is None:
            raise ValueError("RESTART_FROM requires EXTRA_GENERATIONS in the setup file.")
        run_de_restart(de, restart_from, int(extra_generations), seed=int(getattr(module, "SEED", 1000)))
    else:
        run_de(de, seed=int(getattr(module, "SEED", 1000)))

    dry_run = bool(getattr(module, "DRY_RUN", False))
    allow_dry_update = bool(getattr(module, "ALLOW_DRY_RUN_BASEINFO1_UPDATE", False))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)) and (not dry_run or allow_dry_update):
        update_baseinfo1_locidx(de)
    elif dry_run and not allow_dry_update:
        print("Skipping baseinfo1_locidx.csv update because this was a dry run.", flush=True)
