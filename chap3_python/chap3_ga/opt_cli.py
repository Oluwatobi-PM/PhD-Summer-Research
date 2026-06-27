"""Generic optimizer CLI for setup files that choose their optimizer."""

from __future__ import annotations

import argparse

from .optimizer import check_setup, run_from_setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Chapter 3 case using OPTIMIZER from its setup file.")
    parser.add_argument("--setup", required=True, help="Path to a case setup.py/setup_ga.py/setup_ilhs.py file.")
    parser.add_argument("--check-setup", action="store_true", help="Print setup information and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_setup:
        check_setup(args.setup)
    else:
        run_from_setup(args.setup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
