"""Latin-hypercube initialization shared by GA and ILHS experiments."""

from __future__ import annotations

import numpy as np

from .config import CaseConfig
from .encoding import decode_locations, encode_locations


def lhs_initial_chromosomes(cfg: CaseConfig, count: int, seed: int) -> np.ndarray:
    """Generate GA-compatible chromosomes from one LHS design."""

    rng = np.random.default_rng(seed)
    particles, _ = lhs_population(count, normalized_dimension(cfg), rng)
    return decode_lhs_population(cfg, particles)


def lhs_population(
    number_of_samples: int,
    number_of_dimensions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an LHS population and the associated stratum order record."""

    initial = np.zeros((number_of_samples, number_of_dimensions), dtype=float)
    order = np.zeros((number_of_samples, number_of_dimensions), dtype=int)
    for j in range(number_of_dimensions):
        pi_ij = rng.permutation(number_of_samples) + 1
        order[:, j] = pi_ij
        for i in range(number_of_samples):
            random_num = rng.uniform(0.0, 1.0)
            initial[i, j] = (pi_ij[i] - random_num) / number_of_samples
    return initial, order


def decode_lhs_population(cfg: CaseConfig, particles: np.ndarray) -> np.ndarray:
    """Decode normalized LHS samples into the mixed integer chromosome."""

    chrom = np.zeros((particles.shape[0], cfg.chromosome_length), dtype=int)
    for i, particle in enumerate(particles):
        chrom[i] = decode_lhs_particle(cfg, particle)
    return chrom


def decode_lhs_particle(cfg: CaseConfig, particle: np.ndarray) -> np.ndarray:
    """Decode one normalized particle into a GA-compatible chromosome."""

    n = cfg.num_wells
    bpw = cfg.bits_per_location
    chrom = np.zeros(cfg.chromosome_length, dtype=int)
    offset = 0

    if cfg.design_var in (1, 3):
        order_values = particle[offset : offset + n]
        chrom[:n] = rank_order(order_values)
        offset += n

    if cfg.design_var in (1, 2):
        type_values = particle[offset : offset + n]
        type_start = cfg.beforetype * n
        chrom[type_start : type_start + n] = (type_values >= 0.5).astype(int)
        offset += n

        location_values = particle[offset : offset + n]
        locations = decode_unique_locations(location_values, cfg.num_locations)
        encode_locations(locations, cfg.beforeloc, n, bpw, chrom)
        chrom = repair_location_genes(chrom, cfg)

    return chrom


def normalized_dimension(cfg: CaseConfig) -> int:
    """Return the number of normalized LHS variables for a design mode."""

    if cfg.design_var == 1:
        return 3 * cfg.num_wells
    if cfg.design_var == 2:
        return 2 * cfg.num_wells
    if cfg.design_var == 3:
        return cfg.num_wells
    raise ValueError(f"Unsupported design_var: {cfg.design_var}")


def rank_order(values: np.ndarray) -> np.ndarray:
    """Convert normalized order values into unique 1-based drilling ranks."""

    return np.argsort(np.argsort(np.asarray(values, dtype=float))) + 1


def decode_unique_locations(values: np.ndarray, num_locations: int) -> np.ndarray:
    """Map normalized values to unique 1-based candidate locations."""

    raw = np.floor(np.asarray(values, dtype=float) * num_locations).astype(int) + 1
    raw = np.clip(raw, 1, num_locations)
    used: set[int] = set()
    fixed = np.zeros_like(raw)
    for i, loc in enumerate(raw):
        loc = int(loc)
        if loc not in used:
            fixed[i] = loc
            used.add(loc)
            continue
        fixed[i] = nearest_unused_location(loc, used, num_locations)
        used.add(int(fixed[i]))
    return fixed


def nearest_unused_location(target: int, used: set[int], num_locations: int) -> int:
    """Choose the closest available location when LHS creates a duplicate."""

    for radius in range(num_locations):
        low = target - radius
        high = target + radius
        if low >= 1 and low not in used:
            return low
        if high <= num_locations and high not in used:
            return high
    raise RuntimeError("No unused locations remain.")


def repair_location_genes(chrom: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    """Keep encoded location genes in range and unique, without extra randomness."""

    fixed = chrom.copy()
    locs = decode_locations(fixed, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location)
    used: set[int] = set()
    for i, loc in enumerate(locs):
        loc = int(loc)
        if loc < 1 or loc > cfg.num_locations or loc in used:
            loc = nearest_unused_location(max(1, min(loc, cfg.num_locations)), used, cfg.num_locations)
            locs[i] = loc
        used.add(loc)
    encode_locations(locs, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location, fixed)
    return fixed
