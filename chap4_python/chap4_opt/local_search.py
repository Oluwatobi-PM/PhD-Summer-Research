"""Extension points for the chapter 4 StoSAG/GPS local-search routines.

The MATLAB `Iterative` algorithm periodically calls `stosag.m` and `GPSrr.m`.
Those routines are simulator-intensive local optimizers around the current best
well locations. The GA port keeps the iterative chromosome flow intact and
leaves these hooks for a future full local-search translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalSearchResult:
    converged: bool
    locations: np.ndarray
    objective: float
    remaining_locations: np.ndarray
    unselected_locations: np.ndarray


def stosag(locations: np.ndarray, current_objective: float) -> LocalSearchResult:
    """Placeholder for the MATLAB `stosag.m` local-search routine."""

    return LocalSearchResult(False, locations.copy(), current_objective, np.array([]), np.array([]))


def gpsrr(locations: np.ndarray, current_objective: float) -> LocalSearchResult:
    """Placeholder for the MATLAB `GPSrr.m` local-search routine."""

    return LocalSearchResult(False, locations.copy(), current_objective, np.array([]), np.array([]))
