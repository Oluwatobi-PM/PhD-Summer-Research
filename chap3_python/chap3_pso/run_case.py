"""Programmatic runner used by case-level PSO setup files."""

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
from .pso import PSOData, run_pso
from .restart import run_pso_restart


def run_from_setup(setup_file: str | Path) -> None:
    """Run a PSO case directly from its setup file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        print(setup_report(cfg))
        print(f"pso_dimensions: {normalized_dimension(cfg)}")
        print(f"swarm_size: {cfg.population_size}")
        print(f"max_iterations: {cfg.maxgen}")
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
        f"PSO optimization job started: run_id={run_id}, python_pid={os.getpid()}, "
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

    pso = PSOData(
        cfg,
        objective,
        max_iterations=cfg.maxgen,
        swarm_size=cfg.population_size,
        omega=float(getattr(module, "OMEGA", 0.7298)),
        phip=float(getattr(module, "PHIP", 1.496)),
        phig=float(getattr(module, "PHIG", 1.496)),
        velocity_clamp=get_optional_float(module, "VELOCITY_CLAMP"),
        mutation_rate=float(getattr(module, "PSO_MUTATION_RATE", getattr(module, "MUTATION_RATE", 0.0))),
        stall_iterations=get_optional_int(module, "STALL_ITERATIONS"),
        reseed_fraction=float(getattr(module, "RESEED_FRACTION", 0.0)),
        improvement_tolerance=float(getattr(module, "IMPROVEMENT_TOLERANCE", 0.0)),
        initial_particles=initial,
        initial_velocity=str(getattr(module, "INITIAL_VELOCITY", "zero")),
    )
    restart_from = getattr(module, "RESTART_FROM", None)
    if restart_from:
        extra_iterations = getattr(module, "EXTRA_ITERATIONS", getattr(module, "EXTRA_GENERATIONS", None))
        if extra_iterations is None:
            raise ValueError("RESTART_FROM requires EXTRA_ITERATIONS in the setup file.")
        run_pso_restart(pso, restart_from, int(extra_iterations), seed=int(getattr(module, "SEED", 1000)))
    else:
        run_pso(pso, seed=int(getattr(module, "SEED", 1000)))

    dry_run = bool(getattr(module, "DRY_RUN", False))
    allow_dry_update = bool(getattr(module, "ALLOW_DRY_RUN_BASEINFO1_UPDATE", False))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)) and (not dry_run or allow_dry_update):
        update_baseinfo1_locidx(pso)
    elif dry_run and not allow_dry_update:
        print("Skipping baseinfo1_locidx.csv update because this was a dry run.", flush=True)


def get_optional_float(module, name: str) -> float | None:
    value = getattr(module, name, None)
    if value is None:
        return None
    return float(value)


def get_optional_int(module, name: str) -> int | None:
    value = getattr(module, name, None)
    if value is None:
        return None
    return int(value)
