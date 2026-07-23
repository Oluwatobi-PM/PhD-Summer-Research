"""NOMAD/MADS optimizer bridge for Chapter 3 reservoir cases.

This module uses PyNomadBBO/NOMAD for the MADS algorithm and adapts NOMAD's
mixed variable vectors to decoded order/type/location design variables. It
still records chromosome-compatible history for plotting and legacy utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from chap3_ga.checkpoint import atomic_savez, best_population_payload, common_metadata, optimizer_history_dir
from chap3_ga.config import CaseConfig
from chap3_ga.encoding import encode_locations
from chap3_ga.lhs_initialization import nearest_unused_location, rank_order


Objective = Callable[[np.ndarray], np.ndarray]


@dataclass
class MADSVariableSpace:
    """Mixed-variable NOMAD representation for one `CaseConfig`."""

    config: CaseConfig

    def __post_init__(self) -> None:
        self.dimension = variable_dimension(self.config)

    @property
    def lower_bounds(self) -> list[float]:
        lb: list[float] = []
        if self.config.design_var in (1, 3):
            lb.extend([1.0] * self.config.num_wells)
        if self.config.design_var in (1, 2):
            lb.extend([0.0] * self.config.num_wells)
            lb.extend([1.0] * self.config.num_wells)
        return lb

    @property
    def upper_bounds(self) -> list[float]:
        ub: list[float] = []
        if self.config.design_var in (1, 3):
            ub.extend([float(self.config.num_wells)] * self.config.num_wells)
        if self.config.design_var in (1, 2):
            ub.extend([1.0] * self.config.num_wells)
            ub.extend([float(self.config.num_locations)] * self.config.num_wells)
        return ub

    @property
    def input_types(self) -> list[str]:
        types: list[str] = []
        if self.config.design_var in (1, 3):
            types.extend(["I"] * self.config.num_wells)
        if self.config.design_var in (1, 2):
            types.extend(["B"] * self.config.num_wells)
            types.extend(["I"] * self.config.num_wells)
        return types

    def chromosome_from_vector(self, vector: np.ndarray) -> np.ndarray:
        """Convert one NOMAD mixed vector into a compatibility chromosome."""

        design = self.design_from_vector(vector)
        cfg = self.config
        chrom = np.zeros(cfg.chromosome_length, dtype=int)

        if cfg.design_var in (1, 3):
            chrom[: cfg.num_wells] = np.asarray(design["order"], dtype=int)

        if cfg.design_var in (1, 2):
            type_start = cfg.beforetype * cfg.num_wells
            chrom[type_start : type_start + cfg.num_wells] = np.asarray(design["types"], dtype=int)
            locations = np.asarray(design["locations"], dtype=int)
            encode_locations(locations, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location, chrom)

        return chrom

    def design_from_vector(self, vector: np.ndarray) -> dict[str, np.ndarray | None]:
        """Convert one NOMAD mixed vector into decoded design arrays."""

        cfg = self.config
        values = np.asarray(vector, dtype=float).reshape(-1)
        design: dict[str, np.ndarray | None] = {"order": None, "types": None, "locations": None}
        offset = 0

        if cfg.design_var in (1, 3):
            order_values = np.rint(values[offset : offset + cfg.num_wells]).astype(int)
            design["order"] = rank_order(order_values)
            offset += cfg.num_wells

        if cfg.design_var in (1, 2):
            type_values = values[offset : offset + cfg.num_wells]
            design["types"] = np.rint(type_values).astype(int).clip(0, 1)
            offset += cfg.num_wells

            raw_locations = np.rint(values[offset : offset + cfg.num_wells]).astype(int)
            design["locations"] = repair_unique_locations(raw_locations, cfg.num_locations)

        return design

    def vector_from_chromosome(self, chrom: np.ndarray | None = None) -> list[float]:
        """Build a NOMAD start vector from a chromosome or case base info."""

        cfg = self.config
        if chrom is not None:
            return chromosome_to_vector(cfg, np.asarray(chrom, dtype=int))

        values: list[float] = []
        if cfg.design_var in (1, 3):
            values.extend(float(i) for i in range(1, cfg.num_wells + 1))
        if cfg.design_var in (1, 2):
            if cfg.well_type is not None and len(cfg.well_type) >= cfg.num_wells:
                values.extend(float(int(v)) for v in cfg.well_type[: cfg.num_wells])
            else:
                values.extend([1.0] * cfg.num_wells)

            if cfg.locidx is not None and cfg.locidx.size >= cfg.num_wells:
                locs = np.asarray(cfg.locidx).reshape(-1)[: cfg.num_wells]
                values.extend(float(int(v)) for v in locs)
            else:
                values.extend(float(i) for i in range(1, cfg.num_wells + 1))
        return values


@dataclass
class MADSData:
    """Mutable standalone NOMAD/MADS state."""

    config: CaseConfig
    objective: Callable[[list[dict[str, np.ndarray | None]]], np.ndarray]
    max_simulations: int
    initial_mesh_size: float = 0.25
    initial_poll_size: float = 0.25
    min_mesh_size: float = 0.01
    min_poll_size: float = 0.01
    direction_type: str = "ORTHO 2N"
    display_degree: int = 0
    bb_max_block_size: int | None = None
    seed: int | None = None
    x0: list[float] | None = None
    max_iterations: int | None = None
    variable_space: MADSVariableSpace = field(init=False)
    eval_count: int = 0
    simulated_count: int = 0
    cache_hits: int = 0
    xmin: np.ndarray | None = None
    fxmin: float = np.inf
    best_vector: np.ndarray | None = None
    history_vectors: list[np.ndarray] = field(default_factory=list)
    history_designs: list[np.ndarray] = field(default_factory=list)
    history_chrom: list[np.ndarray] = field(default_factory=list)
    history_obj: list[float] = field(default_factory=list)
    xmingen: list[np.ndarray] = field(default_factory=list)
    fxmingen: list[float] = field(default_factory=list)
    cache: dict[tuple, float] = field(default_factory=dict)
    nomad_result: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if int(self.max_simulations) <= 0:
            raise ValueError("max_simulations must be positive.")
        self.variable_space = MADSVariableSpace(self.config)
        if self.bb_max_block_size is None:
            self.bb_max_block_size = max(1, int(self.config.num_parallel))


def run_mads(mads: MADSData, save_history: bool = True) -> MADSData:
    """Run standalone NOMAD/MADS and return the final state."""

    try:
        import PyNomad
    except ImportError as exc:
        raise ImportError("PyNomadBBO is required for OPTIMIZER='mads'. Install it with `pip install PyNomadBBO`.") from exc

    if mads.seed is not None:
        PyNomad.setSeed(int(mads.seed))

    x0 = mads.x0 if mads.x0 is not None else mads.variable_space.vector_from_chromosome()
    params = nomad_parameters(mads)
    callback = make_block_callback(mads, save_history=save_history)
    print(
        f"NOMAD/MADS started: dimension={mads.variable_space.dimension}, "
        f"max_simulations={mads.max_simulations}, block_size={mads.bb_max_block_size}",
        flush=True,
    )
    result = PyNomad.optimize(
        callback,
        x0,
        mads.variable_space.lower_bounds,
        mads.variable_space.upper_bounds,
        params,
    )
    mads.nomad_result = dict(result) if isinstance(result, dict) else {"raw_result": result}
    save_iteration_data(mads)
    print_results(mads)
    return mads


def nomad_parameters(mads: MADSData) -> list[str]:
    """Return NOMAD parameter strings for this case."""

    input_type = "( " + " ".join(mads.variable_space.input_types) + " )"
    has_discrete = any(kind in {"B", "I"} for kind in mads.variable_space.input_types)
    initial_frame = max(float(mads.initial_poll_size), 1.0) if has_discrete else float(mads.initial_poll_size)
    min_frame = max(float(mads.min_poll_size), 1.0) if has_discrete else float(mads.min_poll_size)
    params = [
        f"DIMENSION {mads.variable_space.dimension}",
        "BB_OUTPUT_TYPE OBJ",
        f"BB_INPUT_TYPE {input_type}",
        f"MAX_BB_EVAL {int(mads.max_simulations)}",
        f"BB_MAX_BLOCK_SIZE {int(mads.bb_max_block_size)}",
        f"INITIAL_FRAME_SIZE * {initial_frame}",
        f"MIN_FRAME_SIZE * {min_frame}",
        f"DIRECTION_TYPE {mads.direction_type}",
        f"DISPLAY_DEGREE {int(mads.display_degree)}",
    ]
    if mads.max_iterations is not None:
        params.append(f"MAX_ITERATIONS {int(mads.max_iterations)}")
    return params


def make_block_callback(mads: MADSData, save_history: bool):
    """Create a PyNomad black-box callback for single or block evaluation."""

    def callback(block):
        if hasattr(block, "get_x"):
            return evaluate_block(mads, block, save_history)
        return evaluate_point(mads, block, save_history)

    return callback


def evaluate_block(mads: MADSData, block, save_history: bool) -> list[bool]:
    """Evaluate a NOMAD block using the shared objective in batches."""

    count = int(block.size())
    eval_ok = [False] * count
    remaining = max(0, int(mads.max_simulations) - int(mads.simulated_count))
    pending: list[tuple[int, object, np.ndarray, np.ndarray, tuple[int, ...]]] = []

    for idx in range(count):
        point = block.get_x(idx)
        vector = point_to_array(point)
        design = mads.variable_space.design_from_vector(vector)
        chrom = mads.variable_space.chromosome_from_vector(vector)
        key = design_cache_key(design)
        if key in mads.cache:
            obj = float(mads.cache[key])
            mads.cache_hits += 1
            point.setBBO(str(obj).encode("UTF-8"))
            record_evaluation(mads, vector, design, chrom, obj, save_history)
            eval_ok[idx] = True
        else:
            if len(pending) < remaining:
                pending.append((idx, point, vector, design, chrom, key))

    if pending:
        designs = [item[3] for item in pending]
        objv = np.asarray(mads.objective(designs), dtype=float).reshape(-1)
        mads.simulated_count += int(len(pending))
        for (idx, point, vector, design, chrom, key), obj in zip(pending, objv):
            obj = float(obj)
            mads.cache[key] = obj
            point.setBBO(str(obj).encode("UTF-8"))
            record_evaluation(mads, vector, design, chrom, obj, save_history)
            eval_ok[idx] = True

    return eval_ok


def evaluate_point(mads: MADSData, point, save_history: bool) -> int:
    """Evaluate one NOMAD point."""

    vector = point_to_array(point)
    design = mads.variable_space.design_from_vector(vector)
    chrom = mads.variable_space.chromosome_from_vector(vector)
    key = design_cache_key(design)
    if key in mads.cache:
        obj = float(mads.cache[key])
        mads.cache_hits += 1
    elif int(mads.simulated_count) >= int(mads.max_simulations):
        return 0
    else:
        obj = float(np.asarray(mads.objective([design]), dtype=float)[0])
        mads.cache[key] = obj
        mads.simulated_count += 1
    point.setBBO(str(obj).encode("UTF-8"))
    record_evaluation(mads, vector, design, chrom, obj, save_history)
    return 1


def record_evaluation(
    mads: MADSData,
    vector: np.ndarray,
    design: dict[str, np.ndarray | None],
    chrom: np.ndarray,
    obj: float,
    save_history: bool,
) -> None:
    """Update counters, incumbent, and checkpoint history for one evaluated point."""

    mads.eval_count += 1
    if obj <= float(mads.fxmin):
        mads.fxmin = float(obj)
        mads.best_vector = np.asarray(vector, dtype=float).copy()
        mads.xmin = np.asarray(chrom, dtype=int).copy()
    if mads.xmin is not None:
        mads.xmingen.append(mads.xmin.copy())
        mads.fxmingen.append(float(mads.fxmin))
    if save_history:
        mads.history_vectors.append(np.asarray(vector, dtype=float).copy())
        mads.history_designs.append(design_to_array(design))
        mads.history_chrom.append(np.asarray(chrom, dtype=int).copy())
        mads.history_obj.append(float(obj))
    print(
        f"MADS eval {mads.eval_count}: f={obj}, best={mads.fxmin}, "
        f"simulated={mads.simulated_count}, cache_hits={mads.cache_hits}",
        flush=True,
    )


def point_to_array(point) -> np.ndarray:
    """Convert a PyNomad point to a NumPy vector."""

    return np.array([float(point.get_coord(i)) for i in range(point.size())], dtype=float)


def variable_dimension(cfg: CaseConfig) -> int:
    """Return NOMAD mixed-variable vector size for one design mode."""

    if cfg.design_var == 1:
        return 3 * cfg.num_wells
    if cfg.design_var == 2:
        return 2 * cfg.num_wells
    if cfg.design_var == 3:
        return cfg.num_wells
    raise ValueError(f"Unsupported design_var: {cfg.design_var}")


def repair_unique_locations(raw_locations: np.ndarray, num_locations: int) -> np.ndarray:
    """Clip and repair location labels to be unique and 1-based."""

    raw = np.asarray(raw_locations, dtype=int).reshape(-1)
    fixed = np.zeros_like(raw)
    used: set[int] = set()
    for i, loc in enumerate(raw):
        loc = int(np.clip(loc, 1, num_locations))
        if loc in used:
            loc = nearest_unused_location(loc, used, num_locations)
        fixed[i] = loc
        used.add(loc)
    return fixed


def design_cache_key(design: dict[str, np.ndarray | None]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return a cache key from decoded design arrays."""

    parts: list[tuple[str, tuple[int, ...]]] = []
    for name in ("order", "types", "locations"):
        value = design.get(name)
        if value is not None:
            arr = np.asarray(value, dtype=int).reshape(-1)
            parts.append((name, tuple(int(v) for v in arr)))
    return tuple(parts)


def design_to_array(design: dict[str, np.ndarray | None]) -> np.ndarray:
    """Flatten decoded design arrays for checkpointing."""

    chunks = []
    for name in ("order", "types", "locations"):
        value = design.get(name)
        if value is not None:
            chunks.append(np.asarray(value, dtype=int).reshape(-1))
    if not chunks:
        return np.empty((0,), dtype=int)
    return np.concatenate(chunks).astype(int)


def chromosome_to_vector(cfg: CaseConfig, chrom: np.ndarray) -> list[float]:
    """Convert one chromosome into a NOMAD mixed start vector."""

    values: list[float] = []
    if cfg.design_var in (1, 3):
        values.extend(float(v) for v in chrom[: cfg.num_wells])
    if cfg.design_var in (1, 2):
        type_start = cfg.beforetype * cfg.num_wells
        values.extend(float(v) for v in chrom[type_start : type_start + cfg.num_wells])
        from chap3_ga.encoding import decode_locations

        locs = decode_locations(chrom, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location)
        values.extend(float(v) for v in locs)
    return values


def print_results(mads: MADSData) -> None:
    """Print final NOMAD/MADS result."""

    print("NOMAD/MADS optimization completed", flush=True)
    print(f"   Objective function for xmin: {mads.fxmin}", flush=True)
    print(f"   xmin: {mads.xmin.tolist() if mads.xmin is not None else None}", flush=True)
    print(f"   evaluations seen by callback: {mads.eval_count}", flush=True)
    print(f"   simulator evaluations: {mads.simulated_count}", flush=True)
    print(f"   cache hits: {mads.cache_hits}", flush=True)
    if mads.nomad_result is not None:
        print(f"   NOMAD stop reason: {mads.nomad_result.get('stop_reason')}", flush=True)


def save_iteration_data(mads: MADSData) -> None:
    """Save NOMAD/MADS history to the case work directory."""

    out = optimizer_history_dir(mads.config)
    history_chrom = np.array(mads.history_chrom, dtype=int) if mads.history_chrom else np.empty((0, mads.config.chromosome_length), dtype=int)
    history_obj = np.array(mads.history_obj, dtype=float) if mads.history_obj else np.empty((0,), dtype=float)
    data: dict[str, object] = {
        "method": np.array("mads"),
        "MADSvectors": np.array(mads.history_vectors, dtype=float) if mads.history_vectors else np.empty((0, mads.variable_space.dimension), dtype=float),
        "MADSdesign": np.array(mads.history_designs, dtype=int) if mads.history_designs else np.empty((0, mads.variable_space.dimension), dtype=int),
        "MADSgen": history_chrom,
        "MADSgenb": np.array(mads.xmingen, dtype=int) if mads.xmingen else np.empty((0, mads.config.chromosome_length), dtype=int),
        "MADSobj": history_obj,
        "MADSobjb": -np.array(mads.fxmingen, dtype=float) if mads.fxmingen else np.empty((0,), dtype=float),
        "MADSevalCount": np.array(int(mads.eval_count), dtype=int),
        "MADSsimulatedCount": np.array(int(mads.simulated_count), dtype=int),
        "MADScacheHits": np.array(int(mads.cache_hits), dtype=int),
        "MADSbestVector": np.array([] if mads.best_vector is None else mads.best_vector, dtype=float),
    }
    if mads.nomad_result is not None:
        data["MADSstopReason"] = np.array(str(mads.nomad_result.get("stop_reason", "")))
        data["MADSrunFlag"] = np.array(int(mads.nomad_result.get("run_flag", 999)), dtype=int)
    data.update(
        common_metadata(
            "MADS",
            mads.config,
            mads.eval_count,
            best_chromosome=mads.xmin,
            best_objective=mads.fxmin,
        )
    )
    if history_chrom.size and history_obj.size:
        data.update(best_population_payload(history_chrom, history_obj, prefix="MADS"))
    target = out / "tempdata.npz"
    atomic_savez(target, data, fallback_stem=f"tempdata_mads_eval_{mads.eval_count:04d}", compressed=False)
