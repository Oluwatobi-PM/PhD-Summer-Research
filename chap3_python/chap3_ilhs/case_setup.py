"""Load case-level `setup_ilhs.py` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chap3_ga.case_setup import load_setup_module, resolve_case_path
from chap3_ga.config import CaseConfig, NPVOptions, load_baseinfo


def config_from_setup(path: str | Path) -> CaseConfig:
    """Build `CaseConfig` from an ILHS setup file.

    The shared objective/writer code still expects a `CaseConfig`, but ILHS
    setup files should not need GA-only controls like crossover or mutation.
    """

    module = load_setup_module(path)
    setup_path = Path(path).resolve()
    setup_dir = setup_path.parent
    source_dir = resolve_case_path(setup_dir, getattr(module, "SOURCE_DIR"))
    work_dir_value = getattr(module, "WORK_DIR", None)
    work_dir = resolve_case_path(setup_dir, work_dir_value) if work_dir_value else source_dir
    template_dir_value = getattr(module, "TEMPLATE_DIR", None)
    template_dir = resolve_case_path(setup_dir, template_dir_value) if template_dir_value else None

    num_locations = int(getattr(module, "NUM_LOCATIONS"))
    inj_idx = None
    inj_idx_file = source_dir / "inj_idx.csv"
    if inj_idx_file.exists():
        inj_idx = np.loadtxt(inj_idx_file, delimiter=",", dtype=int).reshape(-1)
    elif hasattr(module, "INJ_IDX_ZERO_START"):
        inj_idx = np.ones(num_locations, dtype=int)
        start = int(getattr(module, "INJ_IDX_ZERO_START"))
        inj_idx[max(start - 1, 0) : num_locations] = 0

    vertidx = None
    if hasattr(module, "VERT_IDX_ZERO_START"):
        vertidx = np.ones(num_locations, dtype=int)
        start = int(getattr(module, "VERT_IDX_ZERO_START"))
        vertidx[max(start - 1, 0) : num_locations] = 0

    cfg = CaseConfig(
        name=getattr(module, "CASE_NAME"),
        source_dir=source_dir,
        work_dir=work_dir,
        template_dir=template_dir,
        design_var=int(getattr(module, "DESIGN_VAR")),
        num_wells=int(getattr(module, "NUM_WELLS")),
        num_locations=num_locations,
        pref=float(getattr(module, "PREF")),
        injref=float(getattr(module, "INJREF")),
        sim_time=float(getattr(module, "SIM_TIME")),
        td=float(getattr(module, "TD")),
        maxgen=int(getattr(module, "MAX_ITERATIONS")),
        population_size=int(getattr(module, "NUMBER_OF_SAMPLES")),
        num_parallel=int(getattr(module, "NUM_PARALLEL")),
        simulation_threads=getattr(module, "SIMULATION_THREADS", None),
        objective_scaling=float(getattr(module, "OBJECTIVE_SCALING", 1.0e9)),
        npv=NPVOptions(
            float(getattr(module, "OIL_PRICE")),
            float(getattr(module, "WATER_PRODUCTION_COST")),
            float(getattr(module, "WATER_INJECTION_COST")),
            float(getattr(module, "DISCOUNT_FACTOR")),
        ),
        cdrill_v=float(getattr(module, "CDRILL_V", 8.0e6)),
        cdrill_h=float(getattr(module, "CDRILL_H", 1.6e7)),
        inj_idx=inj_idx,
        vertidx=vertidx,
    )
    cfg.locidx, cfg.well_type = load_baseinfo(cfg)
    cfg.setup_file = setup_path
    return cfg
