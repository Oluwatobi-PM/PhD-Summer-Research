"""Unified case runner that chooses the optimizer from one setup file."""

from __future__ import annotations

from pathlib import Path

from .case_setup import load_setup_module


def optimizer_name(setup_file: str | Path) -> str:
    """Return the normalized optimizer selected by a setup file."""

    module = load_setup_module(setup_file)
    return str(getattr(module, "OPTIMIZER", "ga")).strip().lower()


def run_from_setup(setup_file: str | Path) -> None:
    """Run the optimizer selected by `OPTIMIZER` in the setup file."""

    selected = optimizer_name(setup_file)
    if selected == "ga":
        from .run_case import run_from_setup as run_ga_from_setup

        run_ga_from_setup(setup_file)
        return
    if selected == "ilhs":
        from chap3_ilhs.run_case import run_from_setup as run_ilhs_from_setup

        run_ilhs_from_setup(setup_file)
        return
    raise ValueError(f"Unsupported OPTIMIZER={selected!r}. Expected 'ga' or 'ilhs'.")


def check_setup(setup_file: str | Path) -> None:
    """Print the selected optimizer's setup report without running CMG."""

    selected = optimizer_name(setup_file)
    if selected == "ga":
        from .case_setup import config_from_setup
        from .config import setup_report

        print(setup_report(config_from_setup(setup_file)))
        return
    if selected == "ilhs":
        from chap3_ilhs.case_setup import config_from_setup
        from chap3_ilhs.ilhs import normalized_dimension
        from .config import setup_report

        cfg = config_from_setup(setup_file)
        print(setup_report(cfg))
        print(f"ilhs_dimensions: {normalized_dimension(cfg)}")
        print(f"number_of_samples: {cfg.population_size}")
        print(f"max_iterations: {cfg.maxgen}")
        return
    raise ValueError(f"Unsupported OPTIMIZER={selected!r}. Expected 'ga' or 'ilhs'.")
