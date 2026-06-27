"""MATLAB-compatible chromosome encoding helpers.

MATLAB's `de2bi`/`bi2de` use little-endian bit order by default. The GA
chromosome stores each candidate well location using that same bit order so
Python-generated chromosomes match the old MATLAB representation.
"""

from __future__ import annotations

import numpy as np


def de2bi(value: int, width: int) -> np.ndarray:
    """MATLAB-compatible little-endian binary encoding."""
    return np.array([(int(value) >> i) & 1 for i in range(width)], dtype=int)


def bi2de(bits: np.ndarray) -> int:
    """MATLAB-compatible little-endian binary decoding."""
    total = 0
    for i, bit in enumerate(np.asarray(bits, dtype=int).reshape(-1)):
        total += int(bit) << i
    return total


def decode_locations(chromosome: np.ndarray, beforeloc: int, num_wells: int, bpw: int) -> np.ndarray:
    """Decode the binary location block into 1-based location labels."""

    chromosome = np.asarray(chromosome, dtype=int)
    start = beforeloc * num_wells
    locs = np.zeros(num_wells, dtype=int)
    for i in range(num_wells):
        lo = start + i * bpw
        locs[i] = bi2de(chromosome[lo : lo + bpw]) + 1
    return locs


def encode_locations(locations: np.ndarray, beforeloc: int, num_wells: int, bpw: int, target: np.ndarray) -> None:
    """Write 1-based location labels back into a chromosome's binary block."""

    start = beforeloc * num_wells
    for i, loc in enumerate(np.asarray(locations, dtype=int).reshape(-1)[:num_wells]):
        lo = start + i * bpw
        target[lo : lo + bpw] = de2bi(int(loc) - 1, bpw)
