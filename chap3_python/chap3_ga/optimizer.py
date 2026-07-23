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
    if selected == "pso":
        from chap3_pso.run_case import run_from_setup as run_pso_from_setup

        run_pso_from_setup(setup_file)
        return
    if selected == "de":
        from chap3_de.run_case import run_from_setup as run_de_from_setup

        run_de_from_setup(setup_file)
        return
    if selected == "mads":
        from chap3_mads.run_case import run_from_setup as run_mads_from_setup

        run_mads_from_setup(setup_file)
        return
    if selected in ("hybrid_mads", "pso_mads", "global_mads"):
        from chap3_hybrid.run_case import run_from_setup as run_hybrid_from_setup

        run_hybrid_from_setup(setup_file)
        return
    raise ValueError(
        f"Unsupported OPTIMIZER={selected!r}. Expected 'ga', 'ilhs', 'pso', 'de', 'mads', or 'hybrid_mads'."
    )


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
    if selected == "pso":
        from chap3_pso.case_setup import config_from_setup
        from chap3_ga.lhs_initialization import normalized_dimension
        from .config import setup_report

        cfg = config_from_setup(setup_file)
        print(setup_report(cfg))
        print(f"pso_dimensions: {normalized_dimension(cfg)}")
        print(f"swarm_size: {cfg.population_size}")
        print(f"max_iterations: {cfg.maxgen}")
        return
    if selected == "de":
        from chap3_de.case_setup import config_from_setup
        from chap3_de.de import STRATEGY_NAMES
        from chap3_ga.lhs_initialization import normalized_dimension
        from .config import setup_report

        module = load_setup_module(setup_file)
        cfg = config_from_setup(setup_file)
        print(setup_report(cfg))
        print(f"de_dimensions: {normalized_dimension(cfg)}")
        print(f"population_size: {cfg.population_size}")
        print(f"max_generations: {cfg.maxgen}")
        print(f"de_strategy: {STRATEGY_NAMES[int(getattr(module, 'DE_STRATEGY', 7))]}")
        return
    if selected == "mads":
        from chap3_mads.case_setup import config_from_setup
        from chap3_mads.mads import MADSVariableSpace
        from .config import setup_report

        cfg = config_from_setup(setup_file)
        space = MADSVariableSpace(cfg)
        print(setup_report(cfg))
        print(f"mads_dimensions: {space.dimension}")
        print(f"max_simulations: {cfg.maxgen}")
        print(f"bb_input_type: {' '.join(space.input_types)}")
        print(f"initial_x0: {space.vector_from_chromosome()}")
        return
    if selected in ("hybrid_mads", "pso_mads", "global_mads"):
        from chap3_hybrid.case_setup import config_from_setup
        from chap3_hybrid.run_case import print_hybrid_options
        from chap3_ga.lhs_initialization import normalized_dimension
        from chap3_mads.mads import MADSVariableSpace
        from .config import setup_report

        module = load_setup_module(setup_file)
        cfg = config_from_setup(setup_file)
        space = MADSVariableSpace(cfg)
        print(setup_report(cfg))
        print(f"global_optimizer: {getattr(module, 'GLOBAL_OPTIMIZER', 'pso')}")
        print(f"pso_dimensions: {normalized_dimension(cfg)}")
        print(f"mads_dimensions: {space.dimension}")
        print(f"bb_input_type: {' '.join(space.input_types)}")
        print_hybrid_options(module, cfg)
        return
    raise ValueError(
        f"Unsupported OPTIMIZER={selected!r}. Expected 'ga', 'ilhs', 'pso', 'de', 'mads', or 'hybrid_mads'."
    )
