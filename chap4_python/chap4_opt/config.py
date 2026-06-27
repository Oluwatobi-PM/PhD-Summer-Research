"""Case setup translated from chapter 4 `setupGA.m` files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

import numpy as np

from .matlab_io import loadmat_arrays


CaseName = Literal["Brugge", "PUNQ"]
AlgorithmName = Literal["MixencodeGA", "GenocopIII", "Iterative"]


@dataclass
class NPVOptions:
    """Economic parameters used to calculate discounted NPV."""

    oil_price: float = 80.0
    water_production_cost: float = 5.0
    water_injection_cost: float = 5.0
    discount_factor: float = 0.1


@dataclass
class CaseConfig:
    """All fixed inputs for one chapter 4 case."""

    name: CaseName
    source_dir: Path
    work_dir: Path
    algorithm: AlgorithmName
    num_inj: int
    num_pro: int
    bpi: int = 2
    bpp: int = 2
    min_rad: float = 3.0
    sim_time: float = 7300.0
    grid_i: int = 0
    grid_j: int = 0
    grid_k: int = 0
    maxgen: int = 1
    population_size: int = 4
    crossover_probability: float = 0.8
    mutation_probability: float = 0.01
    reference_update_probability: float = 0.25
    num_parallel: int = 2
    maxcuts: int = 5
    maxres: int = 4
    maxiter: int = 5
    npert: int = 10
    meshgrid: int = 1
    iter_ga: int = 5
    epsr: float = 1.0e-8
    cdrill: float = 8.0e6
    objective_scaling: float = 1.0e9
    npv: NPVOptions = field(default_factory=NPVOptions)
    locidx: np.ndarray | None = None
    locidx1: np.ndarray | None = None
    oilgb: np.ndarray | None = None
    actnum: np.ndarray | None = None

    @property
    def num_wells(self) -> int:
        return self.num_inj + self.num_pro

    @property
    def combined_length(self) -> int:
        return self.num_inj * (self.bpi + 2) + self.num_pro * (self.bpp + 2)

    @property
    def iterative_length(self) -> int:
        return self.num_inj * self.bpi + self.num_pro * self.bpp


def make_config(
    case: CaseName,
    source_dir: str | Path,
    work_dir: str | Path | None = None,
    algorithm: AlgorithmName | None = None,
) -> CaseConfig:
    """Build a Python config with the constants from the MATLAB setup file."""

    source = Path(source_dir).resolve()
    work = Path(work_dir).resolve() if work_dir else source
    if case == "Brugge":
        cfg = CaseConfig(
            name="Brugge",
            source_dir=source,
            work_dir=work,
            algorithm=algorithm or "GenocopIII",
            num_inj=10,
            num_pro=20,
            grid_i=139,
            grid_j=48,
            grid_k=9,
            cdrill=8.0e6,
        )
    elif case == "PUNQ":
        cfg = CaseConfig(
            name="PUNQ",
            source_dir=source,
            work_dir=work,
            algorithm=algorithm or "MixencodeGA",
            num_inj=10,
            num_pro=10,
            grid_i=19,
            grid_j=28,
            grid_k=5,
            cdrill=30.0e6,
        )
    else:
        raise ValueError(f"Unsupported case: {case}")
    load_case_arrays(cfg)
    return cfg


def load_case_arrays(cfg: CaseConfig) -> None:
    """Load `locidx`, `locidx1`, `oilgb`, and the top-layer `actnum` grid."""

    base = loadmat_arrays(cfg.source_dir / "baseinfo.mat")
    oil = loadmat_arrays(cfg.source_dir / "oilgb.mat")
    cfg.locidx = normalized_matrix(base.get("locidx"))
    cfg.locidx1 = normalized_matrix(base.get("locidx1"))
    cfg.oilgb = normalized_matrix(oil.get("oilgb"))
    actnum_path = cfg.source_dir / "actnum.dat"
    if actnum_path.exists():
        # MATLAB used `fscanf`, which streams every numeric token regardless of
        # line width. `np.loadtxt` expects rectangular rows, so use fromstring.
        tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", actnum_path.read_text())
        values = np.asarray([float(token) for token in tokens], dtype=float)
        expected = cfg.grid_i * cfg.grid_j * cfg.grid_k
        if values.size < expected:
            raise RuntimeError(f"{actnum_path} has {values.size} values; expected {expected}.")
        values = values[:expected].reshape((cfg.grid_i, cfg.grid_j, cfg.grid_k), order="F")
        cfg.actnum = values[:, :, 0]


def normalized_matrix(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr
