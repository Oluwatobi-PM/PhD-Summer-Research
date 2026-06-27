"""Programmatic runner used by case-level `setup_ga.py` files."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import ModuleType

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
    write_optimizer_job_info(cfg.work_dir, run_id, setup_file, module, cfg)
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


def write_optimizer_job_info(
    work_dir: Path,
    run_id: str,
    setup_file: Path,
    module: ModuleType | None = None,
    cfg=None,
) -> None:
    """Save readable run identity and setup metadata for later checking."""

    info_dir = work_dir / "python_tempdata"
    info_dir.mkdir(parents=True, exist_ok=True)
    info = info_dir / "current_job.txt"
    info.write_text("\n".join(job_info_lines(run_id, setup_file, module, cfg)) + "\n")


def job_info_lines(run_id: str, setup_file: Path, module: ModuleType | None, cfg) -> list[str]:
    """Build the contents of `current_job.txt`."""

    lines = [
        "[job]",
        f"run_id = {run_id}",
        f"started_at_local = {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python_pid = {os.getpid()}",
        f"setup_file = {setup_file}",
    ]
    if module is None or cfg is None:
        return lines

    optimizer = str(getattr(module, "OPTIMIZER", "ga")).lower()
    lines.extend(
        [
            "",
            "[case]",
            f"case_name = {cfg.name}",
            f"optimizer = {optimizer}",
            f"design_var = {cfg.design_var} ({design_var_label(cfg.design_var)})",
            f"num_wells = {cfg.num_wells}",
            f"num_locations = {cfg.num_locations}",
            f"source_dir = {cfg.source_dir}",
            f"template_dir = {cfg.template_dir}",
            f"work_dir = {cfg.work_dir}",
            f"data_file = {cfg.loaded_data_file}",
            "",
            "[initialization]",
            f"initialization = {getattr(module, 'INITIALIZATION', 'random')}",
            f"initialization_seed = {getattr(module, 'INITIALIZATION_SEED', '')}",
            f"initialization_size = {getattr(module, 'INITIALIZATION_SIZE', cfg.population_size)}",
            f"optimizer_seed = {getattr(module, 'SEED', '')}",
            "",
            "[run_controls]",
            f"num_parallel = {cfg.num_parallel}",
            f"simulation_threads = {cfg.simulation_threads}",
            f"dry_run = {getattr(module, 'DRY_RUN', False)}",
            f"stream_simulator_output = {getattr(module, 'STREAM_SIMULATOR_OUTPUT', False)}",
            f"print_batch_timing = {getattr(module, 'PRINT_BATCH_TIMING', True)}",
            f"results_timeout_seconds = {getattr(module, 'RESULTS_TIMEOUT_SECONDS', 60.0)}",
            "simulation_interrupt_timeout_seconds = "
            f"{getattr(module, 'SIMULATION_INTERRUPT_TIMEOUT_SECONDS', 60.0)}",
            f"update_baseinfo1_after_run = {getattr(module, 'UPDATE_BASEINFO1_AFTER_RUN', True)}",
        ]
    )

    if optimizer == "ilhs":
        lines.extend(
            [
                "",
                "[ilhs]",
                f"number_of_samples = {cfg.population_size}",
                f"max_iterations = {cfg.maxgen}",
                f"entropy = {getattr(module, 'ENTROPY', '')}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "[ga]",
                f"population_size = {cfg.population_size}",
                f"maxgen = {cfg.maxgen}",
                f"crossover_probability = {cfg.crossover_probability}",
                f"mutation_probability = {cfg.mutation_probability}",
                f"order_mutation_probability = {cfg.order_mutation_probability}",
                f"epsr = {cfg.epsr}",
            ]
        )

    lines.extend(
        [
            "",
            "[model]",
            f"pref = {cfg.pref}",
            f"injref = {cfg.injref}",
            f"sim_time = {cfg.sim_time}",
            f"td = {cfg.td}",
            "",
            "[economics]",
            f"oil_price = {cfg.npv.oil_price}",
            f"water_production_cost = {cfg.npv.water_production_cost}",
            f"water_injection_cost = {cfg.npv.water_injection_cost}",
            f"discount_factor = {cfg.npv.discount_factor}",
            f"cdrill_v = {cfg.cdrill_v}",
            f"cdrill_h = {cfg.cdrill_h}",
            f"objective_scaling = {cfg.objective_scaling}",
        ]
    )
    return lines


def design_var_label(design_var: int) -> str:
    if design_var == 1:
        return "O,T,x combined"
    if design_var == 2:
        return "T,x placement/type"
    if design_var == 3:
        return "O order only"
    return "unknown"


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

    out = Path(cfg.source_dir) / "baseinfo1_locidx.csv"
    np.savetxt(out, updated, delimiter=",", fmt="%d")
    cfg.loaded_data_file = out
    print(f"Updated {out} from the best optimized type/location chromosome.")
