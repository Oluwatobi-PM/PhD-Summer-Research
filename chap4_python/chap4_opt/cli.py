"""Command-line entry point for chapter 4 optimization runs."""

from __future__ import annotations

import argparse

from .config import make_config
from .ga import GAData, run_ga
from .objective import ObjectiveEvaluator, prepare_work_folders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Chapter 4 Python optimization workflow.")
    parser.add_argument("--case", choices=["Brugge", "PUNQ"], required=True)
    parser.add_argument("--source", required=True, help="Original MATLAB/simulator case folder.")
    parser.add_argument("--work-dir", default=None, help="Working folder for generated simulator inputs.")
    parser.add_argument("--algorithm", choices=["MixencodeGA", "GenocopIII", "Iterative"], default=None)
    parser.add_argument("--maxgen", type=int, default=None)
    parser.add_argument("--np", type=int, dest="population_size", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic objective instead of launching simulator.")
    parser.add_argument("--seed", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = make_config(args.case, args.source, args.work_dir, args.algorithm)
    if args.maxgen is not None:
        cfg.maxgen = args.maxgen
    if args.population_size is not None:
        cfg.population_size = args.population_size
    prepare_work_folders(cfg)
    objective = ObjectiveEvaluator(cfg, dry_run=args.dry_run)
    ga = GAData(cfg, objective)
    run_ga(ga, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
