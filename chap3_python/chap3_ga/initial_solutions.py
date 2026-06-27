"""Helpers for seeding GA runs with user-provided initial solutions."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import json

import numpy as np

from .config import CaseConfig
from .encoding import decode_locations, encode_locations


def initial_chromosomes_from_setup(module: ModuleType, cfg: CaseConfig) -> np.ndarray | None:
    """Load optional seeded chromosomes/solutions from a case setup module."""

    raw_chromosomes = getattr(module, "INITIAL_CHROMOSOMES", None)
    structured_solutions = getattr(module, "INITIAL_SOLUTIONS", None)
    if is_empty_seed_value(raw_chromosomes) and is_empty_seed_value(structured_solutions):
        return None

    chromosomes: list[np.ndarray] = []
    if not is_empty_seed_value(raw_chromosomes):
        chromosomes.extend(normalize_raw_chromosomes(raw_chromosomes, cfg))
    if not is_empty_seed_value(structured_solutions):
        chromosomes.extend(solution_to_chromosome(solution, cfg) for solution in structured_solutions)

    return validate_initial_chromosomes(chromosomes, cfg)


def is_empty_seed_value(value: object) -> bool:
    """Return true when a setup seed variable is missing or intentionally empty."""

    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return len(value) == 0  # type: ignore[arg-type]
    except TypeError:
        return False


def initial_chromosomes_from_file(path: str | Path, cfg: CaseConfig) -> np.ndarray:
    """Load raw or structured initial chromosomes from a JSON/CSV/TXT file."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
        if isinstance(data, dict):
            raw = data.get("chromosomes")
            structured = data.get("solutions")
            chromosomes: list[np.ndarray] = []
            if raw is not None:
                chromosomes.extend(normalize_raw_chromosomes(raw, cfg))
            if structured is not None:
                chromosomes.extend(solution_to_chromosome(solution, cfg) for solution in structured)
            return validate_initial_chromosomes(chromosomes, cfg)
        return validate_initial_chromosomes(normalize_raw_chromosomes(data, cfg), cfg)

    arr = np.loadtxt(source, delimiter="," if source.suffix.lower() == ".csv" else None, dtype=int)
    return validate_initial_chromosomes(normalize_raw_chromosomes(arr, cfg), cfg)


def normalize_raw_chromosomes(raw: object, cfg: CaseConfig) -> list[np.ndarray]:
    """Convert one or more raw chromosomes to one-dimensional integer arrays."""

    arr = np.asarray(raw, dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("INITIAL_CHROMOSOMES must be a 1D chromosome or a 2D list of chromosomes.")
    return [row.copy() for row in arr]


def solution_to_chromosome(solution: object, cfg: CaseConfig) -> np.ndarray:
    """Convert a readable solution dictionary into the encoded chromosome."""

    if not isinstance(solution, dict):
        raise ValueError("Each INITIAL_SOLUTIONS entry must be a dictionary.")

    chrom = np.zeros(cfg.chromosome_length, dtype=int)
    if cfg.design_var in (1, 3):
        order = np.asarray(solution.get("order"), dtype=int).reshape(-1)
        validate_order(order, cfg)
        chrom[: cfg.num_wells] = order

    if cfg.design_var in (1, 2):
        types = np.asarray(solution.get("types"), dtype=int).reshape(-1)
        locations = np.asarray(solution.get("locations"), dtype=int).reshape(-1)
        validate_types(types, cfg)
        validate_locations(locations, cfg)
        type_start = cfg.beforetype * cfg.num_wells
        loc_start = cfg.beforeloc * cfg.num_wells
        chrom[type_start:loc_start] = types
        encode_locations(locations, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location, chrom)

    return chrom


def validate_initial_chromosomes(chromosomes: list[np.ndarray], cfg: CaseConfig) -> np.ndarray:
    """Validate shape and basic feasibility of user-seeded chromosomes."""

    if not chromosomes:
        raise ValueError("No initial chromosomes were provided.")
    arr = np.vstack([np.asarray(chrom, dtype=int).reshape(1, -1) for chrom in chromosomes])
    if arr.shape[1] != cfg.chromosome_length:
        raise ValueError(
            f"Initial chromosome length mismatch: expected {cfg.chromosome_length}, got {arr.shape[1]}."
        )
    if cfg.design_var in (1, 3):
        for row in arr:
            validate_order(row[: cfg.num_wells], cfg)
    if cfg.design_var in (1, 2):
        type_start = cfg.beforetype * cfg.num_wells
        loc_start = cfg.beforeloc * cfg.num_wells
        types = arr[:, type_start:loc_start]
        if not np.all((types == 0) | (types == 1)):
            raise ValueError("Initial chromosome type genes must be 0 or 1.")
    return arr


def apply_initial_chromosomes(population: np.ndarray, initial_chromosomes: np.ndarray | None) -> int:
    """Put user-seeded chromosomes in the first rows of a generated population."""

    if initial_chromosomes is None:
        return 0
    seeded = np.asarray(initial_chromosomes, dtype=int)
    if seeded.ndim == 1:
        seeded = seeded.reshape(1, -1)
    if seeded.shape[0] > population.shape[0]:
        raise ValueError(
            f"Received {seeded.shape[0]} initial chromosome(s), but population size is only {population.shape[0]}."
        )
    if seeded.shape[1] != population.shape[1]:
        raise ValueError(
            f"Initial chromosome length mismatch: expected {population.shape[1]}, got {seeded.shape[1]}."
        )
    population[: seeded.shape[0], :] = seeded
    return int(seeded.shape[0])


def describe_initial_chromosomes(chromosomes: np.ndarray, cfg: CaseConfig) -> str:
    """Return a readable summary of how seeded chromosomes will be interpreted."""

    seeded = np.asarray(chromosomes, dtype=int)
    if seeded.ndim == 1:
        seeded = seeded.reshape(1, -1)
    lines: list[str] = []
    for idx, chrom in enumerate(seeded, start=1):
        lines.append(f"Seeded chromosome {idx}:")
        if cfg.design_var in (1, 2):
            type_start = cfg.beforetype * cfg.num_wells
            loc_start = cfg.beforeloc * cfg.num_wells
            types = chrom[type_start:loc_start]
            locations = decode_locations(chrom, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location)
            forced = np.array([is_forced_injector_location(cfg, loc) for loc in locations], dtype=bool)
            effective = (types.astype(bool) | forced)
            lines.append(f"  locations: {locations.astype(int).tolist()}")
            lines.append(f"  type bits: {types.astype(int).tolist()} (1=injector, 0=producer)")
            lines.append(f"  forced injector flags: {forced.astype(int).tolist()}")
            lines.append(
                f"  effective count: {int(effective.sum())} injector(s), "
                f"{int(cfg.num_wells - effective.sum())} producer(s)"
            )
        if cfg.design_var in (1, 3):
            order = chrom[: cfg.num_wells]
            lines.append(f"  order: {order.astype(int).tolist()}")
        if cfg.design_var == 3 and cfg.locidx is not None:
            active_rows = np.asarray([row for row in cfg.locidx if row.shape[0] >= 4 and row[3] >= 0])
            if active_rows.size:
                types = active_rows[:, 3].astype(int)
                lines.append(f"  fixed active types from baseinfo1: {types.tolist()} (1=injector, 0=producer)")
                lines.append(
                    f"  fixed count: {int(np.sum(types == 1))} injector(s), "
                    f"{int(np.sum(types == 0))} producer(s)"
                )
    return "\n".join(lines)


def is_forced_injector_location(cfg: CaseConfig, loc_number_1based: int) -> bool:
    """Return true when a selected candidate location is forced to injector."""

    if cfg.inj_idx is None:
        return False
    idx = int(loc_number_1based) - 1
    return 0 <= idx < len(cfg.inj_idx) and bool(cfg.inj_idx[idx])


def validate_order(order: np.ndarray, cfg: CaseConfig) -> None:
    expected = np.arange(1, cfg.num_wells + 1)
    if order.shape[0] != cfg.num_wells or not np.array_equal(np.sort(order), expected):
        raise ValueError(f"Order genes must be a permutation of 1..{cfg.num_wells}.")


def validate_types(types: np.ndarray, cfg: CaseConfig) -> None:
    if types.shape[0] != cfg.num_wells:
        raise ValueError(f"Expected {cfg.num_wells} type values.")
    if not np.all((types == 0) | (types == 1)):
        raise ValueError("Type values must be 0 for producer or 1 for injector.")


def validate_locations(locations: np.ndarray, cfg: CaseConfig) -> None:
    if locations.shape[0] != cfg.num_wells:
        raise ValueError(f"Expected {cfg.num_wells} location values.")
    if np.any(locations < 1) or np.any(locations > cfg.num_locations):
        raise ValueError(f"Location values must be between 1 and {cfg.num_locations}.")
    if len(set(int(x) for x in locations)) != cfg.num_wells:
        raise ValueError("Location values must be unique within one chromosome.")
