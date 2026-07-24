"""Programmatic runner used by case-level NOMAD/MADS setup files."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from chap3_ga.case_setup import load_setup_module
from chap3_ga.config import setup_report
from chap3_ga.objective import DesignPopulationEvaluator, ObjectiveEvaluator, clean_generated_work_folders, prepare_work_folders
from chap3_ga.run_case import update_baseinfo1_locidx, write_optimizer_job_info

from .case_setup import config_from_setup
from .mads import MADSData, MADSVariableSpace, design_cache_key, run_mads


def run_from_setup(setup_file: str | Path) -> None:
    """Run a NOMAD/MADS case directly from its setup file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)
    space = MADSVariableSpace(cfg)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        initial_chromosome = initial_chromosome_from_setup(module, cfg)
        initial_x0 = (
            space.vector_from_chromosome(initial_chromosome)
            if initial_chromosome is not None
            else space.vector_from_chromosome()
        )
        print(setup_report(cfg))
        print(f"mads_dimensions: {space.dimension}")
        print(f"max_simulations: {cfg.maxgen}")
        print(f"bb_input_type: {' '.join(space.input_types)}")
        print(f"initial_x0: {initial_x0}")
        return

    if bool(getattr(module, "CLEAN_WORK_FOLDERS_ON_START", True)):
        clean_generated_work_folders(
            cfg,
            clean_history=bool(getattr(module, "CLEAN_HISTORY_ON_START", True)),
        )
    prepare_work_folders(cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_optimizer_job_info(cfg.work_dir, run_id, setup_file, module, cfg)
    print(
        f"NOMAD/MADS optimization job started: run_id={run_id}, python_pid={os.getpid()}, "
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
    design_objective = DesignPopulationEvaluator(objective)

    max_simulations = getattr(module, "MAX_SIMULATIONS", None)
    if max_simulations is None:
        max_simulations = getattr(module, "MAX_BB_EVAL")
    initial_chromosome = initial_chromosome_from_setup(module, cfg)

    mads = MADSData(
        cfg,
        design_objective,
        max_simulations=int(max_simulations),
        initial_mesh_size=float(getattr(module, "INITIAL_MESH_SIZE", 0.25)),
        initial_poll_size=float(getattr(module, "INITIAL_POLL_SIZE", 0.25)),
        min_mesh_size=float(getattr(module, "MIN_MESH_SIZE", 0.01)),
        min_poll_size=float(getattr(module, "MIN_POLL_SIZE", 0.01)),
        direction_type=str(getattr(module, "DIRECTION_TYPE", "ORTHO 2N")),
        display_degree=int(getattr(module, "DISPLAY_DEGREE", 0)),
        bb_max_block_size=int(getattr(module, "BB_MAX_BLOCK_SIZE", cfg.num_parallel)),
        seed=int(getattr(module, "SEED", 1000)),
        x0=space.vector_from_chromosome(initial_chromosome) if initial_chromosome is not None else None,
    )
    seed_initial_objective(mads, module, initial_chromosome)
    run_mads(mads)

    dry_run = bool(getattr(module, "DRY_RUN", False))
    allow_dry_update = bool(getattr(module, "ALLOW_DRY_RUN_BASEINFO1_UPDATE", False))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)) and (not dry_run or allow_dry_update):
        update_baseinfo1_locidx(mads)
    elif dry_run and not allow_dry_update:
        print("Skipping baseinfo1_locidx.csv update because this was a dry run.", flush=True)


def initial_chromosome_from_setup(module, cfg) -> np.ndarray | None:
    """Return an optional setup-provided starting chromosome for MADS."""

    value = getattr(module, "INITIAL_CHROMOSOME", None)
    if value is None:
        return None
    chrom = np.asarray(value, dtype=int).reshape(-1)
    expected = int(cfg.chromosome_length)
    if chrom.size != expected:
        raise ValueError(f"INITIAL_CHROMOSOME has length {chrom.size}; expected {expected}.")
    return chrom


def seed_initial_objective(mads: MADSData, module, initial_chromosome: np.ndarray | None) -> None:
    """Seed the MADS cache with a known objective for the supplied start point."""

    if initial_chromosome is None or not hasattr(module, "INITIAL_OBJECTIVE"):
        return
    if mads.x0 is None:
        return
    objective = float(getattr(module, "INITIAL_OBJECTIVE"))
    vector = np.asarray(mads.x0, dtype=float)
    design = mads.variable_space.design_from_vector(vector)
    mads.cache[design_cache_key(design)] = objective
    mads.fxmin = objective
    mads.xmin = np.asarray(initial_chromosome, dtype=int).copy()
    mads.best_vector = vector.copy()
