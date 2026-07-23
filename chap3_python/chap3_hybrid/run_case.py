"""Programmatic runner for hybrid global-search + NOMAD/MADS setup files."""

from __future__ import annotations

import os
import time
from pathlib import Path

from chap3_ga.case_setup import load_setup_module
from chap3_ga.config import setup_report
from chap3_ga.lhs_initialization import normalized_dimension
from chap3_ga.objective import ObjectiveEvaluator, clean_generated_work_folders, prepare_work_folders
from chap3_ga.run_case import update_baseinfo1_locidx, write_optimizer_job_info
from chap3_mads.mads import MADSVariableSpace

from .case_setup import config_from_setup
from .global_mads import HybridMADSData, run_hybrid_mads


def run_from_setup(setup_file: str | Path) -> None:
    """Run a hybrid optimizer case directly from its setup file."""

    setup_file = Path(setup_file).resolve()
    module = load_setup_module(setup_file)
    cfg = config_from_setup(setup_file)
    space = MADSVariableSpace(cfg)

    if bool(getattr(module, "CHECK_SETUP_ONLY", False)):
        print(setup_report(cfg))
        print(f"pso_dimensions: {normalized_dimension(cfg)}")
        print(f"mads_dimensions: {space.dimension}")
        print(f"bb_input_type: {' '.join(space.input_types)}")
        print_hybrid_options(module, cfg)
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
        f"Hybrid global-search/MADS job started: run_id={run_id}, python_pid={os.getpid()}, "
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

    hybrid = HybridMADSData(
        cfg,
        objective,
        global_optimizer=str(getattr(module, "GLOBAL_OPTIMIZER", "pso")),
        max_simulations=int(getattr(module, "MAX_SIMULATIONS", cfg.maxgen)),
        pso_handoff_rule=str(getattr(module, "PSO_HANDOFF_RULE", "no_improvement")),
        pso_stall_iterations=int(getattr(module, "PSO_STALL_ITERATIONS", 1)),
        global_iterations_per_cycle=int(getattr(module, "GLOBAL_ITERATIONS_PER_CYCLE", 2)),
        local_mads_budget=int(getattr(module, "LOCAL_MADS_BUDGET", 100)),
        mads_iterations_per_episode=int(getattr(module, "MADS_ITERATIONS_PER_EPISODE", 1)),
        mads_frame_reduction=float(getattr(module, "MADS_FRAME_REDUCTION", 0.5)),
        initial_poll_size=float(getattr(module, "INITIAL_POLL_SIZE", 1.0)),
        min_poll_size=float(getattr(module, "MIN_POLL_SIZE", 1.0)),
        direction_type=str(getattr(module, "DIRECTION_TYPE", "ORTHO 2N")),
        display_degree=int(getattr(module, "DISPLAY_DEGREE", 0)),
        bb_max_block_size=int(getattr(module, "BB_MAX_BLOCK_SIZE", cfg.num_parallel)),
        seed=int(getattr(module, "SEED", 1000)),
        omega=float(getattr(module, "OMEGA", 0.7298)),
        phip=float(getattr(module, "PHIP", 1.496)),
        phig=float(getattr(module, "PHIG", 1.496)),
        velocity_clamp=get_optional_float(module, "VELOCITY_CLAMP"),
        mutation_rate=float(getattr(module, "PSO_MUTATION_RATE", getattr(module, "MUTATION_RATE", 0.0))),
        improvement_tolerance=float(getattr(module, "IMPROVEMENT_TOLERANCE", 0.0)),
        initialization=str(getattr(module, "INITIALIZATION", "lhs")),
        initialization_seed=get_optional_int(module, "INITIALIZATION_SEED"),
        initial_velocity=str(getattr(module, "INITIAL_VELOCITY", "zero")),
    )
    run_hybrid_mads(hybrid)

    dry_run = bool(getattr(module, "DRY_RUN", False))
    allow_dry_update = bool(getattr(module, "ALLOW_DRY_RUN_BASEINFO1_UPDATE", False))
    if bool(getattr(module, "UPDATE_BASEINFO1_AFTER_RUN", True)) and (not dry_run or allow_dry_update):
        update_baseinfo1_locidx(hybrid)
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


def hybrid_option_values(module, cfg) -> dict[str, object]:
    """Return the supported HYBRID_OPTIONS keys and active values."""

    return {
        "GLOBAL_OPTIMIZER": getattr(module, "GLOBAL_OPTIMIZER", "pso"),
        "MAX_SIMULATIONS": getattr(module, "MAX_SIMULATIONS", cfg.maxgen),
        "PSO_HANDOFF_RULE": getattr(module, "PSO_HANDOFF_RULE", "no_improvement"),
        "PSO_STALL_ITERATIONS": getattr(module, "PSO_STALL_ITERATIONS", 1),
        "GLOBAL_ITERATIONS_PER_CYCLE": getattr(module, "GLOBAL_ITERATIONS_PER_CYCLE", 2),
        "LOCAL_MADS_BUDGET": getattr(module, "LOCAL_MADS_BUDGET", 100),
        "MADS_ITERATIONS_PER_EPISODE": getattr(module, "MADS_ITERATIONS_PER_EPISODE", 1),
        "MADS_FRAME_REDUCTION": getattr(module, "MADS_FRAME_REDUCTION", 0.5),
        "INITIAL_POLL_SIZE": getattr(module, "INITIAL_POLL_SIZE", 1.0),
        "MIN_POLL_SIZE": getattr(module, "MIN_POLL_SIZE", 1.0),
        "DIRECTION_TYPE": getattr(module, "DIRECTION_TYPE", "ORTHO 2N"),
        "DISPLAY_DEGREE": getattr(module, "DISPLAY_DEGREE", 0),
        "BB_MAX_BLOCK_SIZE": getattr(module, "BB_MAX_BLOCK_SIZE", cfg.num_parallel),
        "SEED": getattr(module, "SEED", 1000),
        "OMEGA": getattr(module, "OMEGA", 0.7298),
        "PHIP": getattr(module, "PHIP", 1.496),
        "PHIG": getattr(module, "PHIG", 1.496),
        "VELOCITY_CLAMP": getattr(module, "VELOCITY_CLAMP", None),
        "PSO_MUTATION_RATE": getattr(module, "PSO_MUTATION_RATE", getattr(module, "MUTATION_RATE", 0.0)),
        "IMPROVEMENT_TOLERANCE": getattr(module, "IMPROVEMENT_TOLERANCE", 0.0),
        "INITIAL_VELOCITY": getattr(module, "INITIAL_VELOCITY", "zero"),
    }


def print_hybrid_options(module, cfg) -> None:
    """Print supported hybrid option keys and their active values."""

    print("hybrid_options:")
    for key, value in hybrid_option_values(module, cfg).items():
        print(f"  {key}: {value}")
