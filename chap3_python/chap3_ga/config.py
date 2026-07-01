"""Case configuration and MATLAB data loading.

The original MATLAB code stored most run settings as globals in `setupGA.m`.
This module keeps those values in a `CaseConfig` object so the rest of the
Python code can pass configuration explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import struct
import zlib

import numpy as np


CaseName = Literal["Brugge_CaseA", "Brugge_CaseA_xt_o", "channelmodel"]


@dataclass
class NPVOptions:
    """Economic parameters used when converting CMG production output to NPV."""

    oil_price: float
    water_production_cost: float
    water_injection_cost: float
    discount_factor: float


@dataclass
class CaseConfig:
    """All fixed inputs for one optimization case.

    `source_dir` points to the permanent case input library. It can also be
    used as the run-folder template. `work_dir` is where Python writes
    generated numbered simulations and optimizer history.
    """

    name: CaseName
    source_dir: Path
    work_dir: Path
    template_dir: Path | None = None
    design_var: int = 1
    num_wells: int = 0
    num_locations: int = 30
    pref: float = 0.0
    injref: float = 0.0
    sim_time: float = 0.0
    td: float = 0.0
    maxgen: int = 0
    population_size: int = 50
    crossover_probability: float = 0.9
    mutation_probability: float = 0.01
    order_mutation_probability: float = 0.03
    epsr: float = 1.0e-8
    num_parallel: int = 1
    simulation_threads: int | None = None
    objective_scaling: float = 1.0e9
    npv: NPVOptions = field(default_factory=lambda: NPVOptions(80.0, 5.0, 5.0, 0.1))
    cdrill_v: float = 8.0e6
    cdrill_h: float = 1.6e7
    inj_idx: np.ndarray | None = None
    vertidx: np.ndarray | None = None
    locidx: np.ndarray | None = None
    well_type: np.ndarray | None = None
    loaded_data_file: Path | None = None
    setup_file: Path | None = None

    @property
    def bits_per_location(self) -> int:
        return int(np.ceil(np.log2(self.num_locations)))

    @property
    def beforeloc(self) -> int:
        """Number of chromosome blocks before the binary location block."""

        if self.design_var == 1:
            return 2
        if self.design_var == 2:
            return 1
        return 0

    @property
    def beforetype(self) -> int:
        """Number of chromosome blocks before the well type block."""

        return 1 if self.design_var == 1 else 0

    @property
    def bits_per_well(self) -> int:
        if self.design_var == 1:
            return 2 + self.bits_per_location
        if self.design_var == 2:
            return 1 + self.bits_per_location
        return 1

    @property
    def chromosome_length(self) -> int:
        return self.bits_per_well * self.num_wells


def load_baseinfo(cfg: CaseConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load case-specific location/type tables from `baseinfo*.mat`."""

    data_stem = "baseinfo1" if cfg.design_var == 3 and cfg.name != "channelmodel" else "baseinfo"
    csv_locidx = cfg.source_dir / f"{data_stem}_locidx.csv"
    csv_type = cfg.source_dir / f"{data_stem}_type.csv"
    if csv_locidx.exists():
        cfg.loaded_data_file = csv_locidx
        locidx = np.loadtxt(csv_locidx, delimiter=",")
        if locidx.ndim == 1:
            locidx = locidx.reshape(1, -1)
        well_type = np.loadtxt(csv_type, delimiter=",") if csv_type.exists() else None
        if well_type is not None:
            well_type = np.asarray(well_type).reshape(-1)
        return locidx, well_type

    data_path = cfg.loaded_data_file or cfg.source_dir / f"{data_stem}.mat"
    cfg.loaded_data_file = data_path
    if not data_path.exists():
        return None, None
    data = loadmat_arrays(data_path)
    locidx = data.get("locidx")
    well_type = data.get("type")
    if locidx is not None:
        locidx = np.asarray(locidx)
        if locidx.ndim == 1:
            locidx = locidx.reshape(-1, 1)
    if well_type is not None:
        well_type = np.asarray(well_type).reshape(-1)
    return locidx, well_type


def setup_report(cfg: CaseConfig) -> str:
    """Return a human-readable setup report for one configured case."""

    lines = [
        f"case: {cfg.name}",
        f"source_dir: {cfg.source_dir}",
        f"work_dir: {cfg.work_dir}",
        f"template_dir: {cfg.template_dir}",
        f"design_var: {cfg.design_var}",
        f"num_wells: {cfg.num_wells}",
        f"num_locations: {cfg.num_locations}",
        f"chromosome_length: {cfg.chromosome_length}",
        f"bits_per_location: {cfg.bits_per_location}",
        f"num_parallel: {cfg.num_parallel}",
        f"simulation_threads: {cfg.simulation_threads}",
        f"data file selected: {cfg.loaded_data_file}",
        f"data file exists: {cfg.loaded_data_file.exists() if cfg.loaded_data_file else False}",
        f"locidx shape: {None if cfg.locidx is None else cfg.locidx.shape}",
        f"type shape: {None if cfg.well_type is None else cfg.well_type.shape}",
        f"inj_idx shape: {None if cfg.inj_idx is None else cfg.inj_idx.shape}",
    ]
    if cfg.setup_file is not None:
        lines.insert(0, f"setup_file: {cfg.setup_file}")
    warnings = validate_setup(cfg)
    if warnings:
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("status: setup looks ready")
    return "\n".join(lines)


def validate_setup(cfg: CaseConfig) -> list[str]:
    """Check that required case data is available for the active design mode."""

    warnings: list[str] = []
    if cfg.loaded_data_file is None or not cfg.loaded_data_file.exists():
        warnings.append("required data file is missing")
    if cfg.design_var in (1, 2) and cfg.name != "channelmodel" and cfg.locidx is None:
        warnings.append("Brugge design_var 1/2 requires locidx from baseinfo.mat")
    if cfg.design_var == 3 and cfg.locidx is None:
        warnings.append("design_var 3 requires locidx from baseinfo1.mat/baseinfo.mat")
    if cfg.design_var == 3 and cfg.name == "channelmodel" and cfg.well_type is None:
        warnings.append("channelmodel design_var 3 requires type from baseinfo.mat/baseinfo_type.csv")
    if (
        cfg.design_var in (1, 2)
        and cfg.name != "channelmodel"
        and cfg.locidx is not None
        and cfg.locidx.shape[0] < cfg.num_locations
    ):
        warnings.append("locidx has fewer rows than num_locations")
    if cfg.template_dir is None or not cfg.template_dir.exists():
        warnings.append("template_dir is missing")
    return warnings


def loadmat_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load numeric arrays from a MATLAB v5 MAT-file.

    This covers the small `baseinfo*.mat` files used by the converted
    workflow and avoids making SciPy mandatory for running existing cases.
    """
    try:
        from scipy.io import loadmat  # type: ignore

        return {k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")}
    except ImportError:
        pass

    raw = path.read_bytes()
    if len(raw) < 128:
        return {}
    endian = "<" if raw[126:128] == b"IM" else ">"
    arrays: dict[str, np.ndarray] = {}
    offset = 128
    while offset + 8 <= len(raw):
        dtype, nbytes, offset = read_tag(raw, offset, endian)
        payload = raw[offset : offset + nbytes]
        offset += pad8(nbytes)
        if dtype == 15:  # miCOMPRESSED
            parse_mat_stream(zlib.decompress(payload), endian, arrays)
        elif dtype == 14:  # miMATRIX
            item = parse_matrix(payload, endian)
            if item is not None:
                name, value = item
                arrays[name] = value
    return arrays


def parse_mat_stream(raw: bytes, endian: str, arrays: dict[str, np.ndarray]) -> None:
    offset = 0
    while offset + 8 <= len(raw):
        dtype, nbytes, offset = read_tag(raw, offset, endian)
        payload = raw[offset : offset + nbytes]
        offset += pad8(nbytes)
        if dtype == 14:
            item = parse_matrix(payload, endian)
            if item is not None:
                name, value = item
                arrays[name] = value


def parse_matrix(payload: bytes, endian: str) -> tuple[str, np.ndarray] | None:
    """Parse one numeric miMATRIX payload from a MATLAB v5 MAT-file."""

    offset = 0
    _, nbytes, offset = read_tag(payload, offset, endian)  # array flags
    offset += pad8(nbytes)
    _, nbytes, offset = read_tag(payload, offset, endian)  # dimensions
    dims_raw = payload[offset : offset + nbytes]
    offset += pad8(nbytes)
    dims = struct.unpack(endian + "i" * (nbytes // 4), dims_raw)
    _, nbytes, offset = read_tag(payload, offset, endian)  # name
    name = payload[offset : offset + nbytes].decode("latin1")
    offset += pad8(nbytes)
    dtype, nbytes, offset = read_tag(payload, offset, endian)  # real data
    data = payload[offset : offset + nbytes]
    if dtype == 9:  # miDOUBLE
        values = np.frombuffer(data, dtype=endian + "f8")
    elif dtype == 7:  # miSINGLE
        values = np.frombuffer(data, dtype=endian + "f4")
    elif dtype == 5:  # miINT32
        values = np.frombuffer(data, dtype=endian + "i4")
    elif dtype == 4:  # miUINT16
        values = np.frombuffer(data, dtype=endian + "u2")
    elif dtype == 3:  # miINT16
        values = np.frombuffer(data, dtype=endian + "i2")
    elif dtype == 2:  # miUINT8
        values = np.frombuffer(data, dtype="u1")
    elif dtype == 1:  # miINT8
        values = np.frombuffer(data, dtype="i1")
    else:
        return None
    shape = tuple(int(d) for d in dims)
    if not shape:
        return name, values.copy()
    return name, values.reshape(shape, order="F").squeeze().copy()


def read_tag(raw: bytes, offset: int, endian: str) -> tuple[int, int, int]:
    """Read a MATLAB v5 data element tag, including the small-data format."""

    word1, word2 = struct.unpack_from(endian + "II", raw, offset)
    offset += 8
    dtype = word1 & 0xFFFF
    small_nbytes = (word1 >> 16) & 0xFFFF
    if small_nbytes:
        nbytes = small_nbytes
        offset -= 4
    else:
        dtype = word1
        nbytes = word2
    return dtype, nbytes, offset


def pad8(nbytes: int) -> int:
    return nbytes + ((8 - nbytes % 8) % 8)
