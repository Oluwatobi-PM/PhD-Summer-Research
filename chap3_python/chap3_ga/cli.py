"""Command-line entry point for running one converted Chapter 3 case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import load_baseinfo, setup_report
from .case_setup import config_from_setup, load_setup_module
from .ga import GAData, run_ga
from .initial_solutions import initial_chromosomes_from_file, initial_chromosomes_from_setup
from .objective import ObjectiveEvaluator, prepare_work_folders
from .restart import run_ga_restart
from .run_case import write_optimizer_job_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Chapter 3 Python GA/CMG workflow.")
    parser.add_argument("--setup", required=True, help="Path to a case setup_ga.py file.")
    parser.add_argument("--work-dir", default=None, help="Working folder for generated simulator inputs.")
    parser.add_argument("--design-var", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--maxgen", type=int, default=None)
    parser.add_argument("--np", type=int, dest="population_size", default=None)
    parser.add_argument("--num-parallel", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic objective instead of launching CMG.")
    parser.add_argument(
        "--initial-chromosomes",
        default=None,
        help="Optional JSON/CSV/TXT file containing user-seeded initial chromosomes.",
    )
    parser.add_argument("--restart-from", default=None, help="Path to an existing tempdata.npz restart checkpoint.")
    parser.add_argument("--extra-generations", type=int, default=None, help="Number of generations to run after restart.")
    parser.add_argument("--check-setup", action="store_true", help="Print loaded MAT files and case setup, then exit.")
    parser.add_argument("--seed", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = load_setup_module(args.setup)
    cfg = config_from_setup(args.setup)
    if args.work_dir is not None:
        cfg.work_dir = Path(args.work_dir).resolve()
        setattr(module, "WORK_DIR", str(cfg.work_dir))
    if args.design_var is not None:
        cfg.design_var = args.design_var
        setattr(module, "DESIGN_VAR", cfg.design_var)
        cfg.loaded_data_file = None
        cfg.locidx, cfg.well_type = load_baseinfo(cfg)
    if args.maxgen is not None:
        cfg.maxgen = args.maxgen
        setattr(module, "MAXGEN", cfg.maxgen)
    if args.population_size is not None:
        cfg.population_size = args.population_size
        setattr(module, "POPULATION_SIZE", cfg.population_size)
        setattr(module, "INITIALIZATION_SIZE", cfg.population_size)
    if args.num_parallel is not None:
        cfg.num_parallel = args.num_parallel
        setattr(module, "NUM_PARALLEL", cfg.num_parallel)
    if args.check_setup:
        print(setup_report(cfg))
        return 0
    if args.dry_run:
        cfg.num_parallel = min(cfg.num_parallel, cfg.population_size)
        setattr(module, "DRY_RUN", True)
        setattr(module, "NUM_PARALLEL", cfg.num_parallel)
    prepare_work_folders(cfg)
    if args.restart_from is not None:
        setattr(module, "RESTART_FROM", args.restart_from)
    if args.extra_generations is not None:
        setattr(module, "EXTRA_GENERATIONS", args.extra_generations)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    write_optimizer_job_info(cfg.work_dir, run_id, Path(args.setup).resolve(), module, cfg)
    objective = ObjectiveEvaluator(cfg, dry_run=args.dry_run)
    initial_chromosomes = (
        initial_chromosomes_from_file(args.initial_chromosomes, cfg)
        if args.initial_chromosomes is not None
        else initial_chromosomes_from_setup(module, cfg)
    )
    ga = GAData(cfg, objective, initial_chromosomes=initial_chromosomes)
    if args.restart_from is not None:
        if args.extra_generations is None:
            raise SystemExit("--restart-from requires --extra-generations.")
        run_ga_restart(ga, args.restart_from, args.extra_generations, seed=args.seed)
    else:
        run_ga(ga, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
