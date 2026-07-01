"""CMG include-file writers translated from `writeloc*.m` and `writesch*.m`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CaseConfig


def write_case_inputs(cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray) -> None:
    """Dispatch to the writer pair required by the active design variable."""

    if cfg.design_var == 1:
        write_locations_type_location(cfg, fileid, chromosome, loc_indices)
        write_schedule_order_type_location(cfg, fileid, chromosome, loc_indices)
    elif cfg.design_var == 2:
        write_locations_type_location(cfg, fileid, chromosome, loc_indices)
        write_schedule_type_location(cfg, fileid, chromosome, loc_indices)
    elif cfg.design_var == 3:
        write_locations_order_only(cfg, fileid)
        write_schedule_order_only(cfg, fileid, chromosome)
    else:
        raise ValueError(f"Unsupported design_var: {cfg.design_var}")


def case_folder(cfg: CaseConfig, fileid: int) -> Path:
    return Path(cfg.work_dir) / str(fileid)


def well_name(label: int, is_injector: bool) -> str:
    """Format CMG well names such as `'I03'` or `'P12'`."""

    prefix = "I" if is_injector else "P"
    return f"'{prefix}{int(label):02d}'"


def is_forced_injector(cfg: CaseConfig, loc_number_1based: int) -> bool:
    """Return true for Brugge aquifer locations that MATLAB forced to inject."""

    if cfg.inj_idx is None:
        return False
    idx = int(loc_number_1based) - 1
    return 0 <= idx < len(cfg.inj_idx) and bool(cfg.inj_idx[idx])


def selected_type(cfg: CaseConfig, chromosome: np.ndarray, well_idx: int) -> bool:
    """Read one well's type bit from the chromosome."""

    return bool(chromosome[cfg.beforetype * cfg.num_wells + well_idx])


def write_locations_type_location(
    cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray
) -> None:
    if cfg.name == "channelmodel":
        write_channel_locations(cfg, fileid, chromosome, loc_indices)
    else:
        write_brugge_locations(cfg, fileid, chromosome, loc_indices)


def write_brugge_locations(cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray) -> None:
    """Write Brugge `waterFlooding_well_location.inc`."""

    if cfg.locidx is None:
        raise RuntimeError("Brugge location writer needs locidx from baseinfo.mat.")
    path = case_folder(cfg, fileid) / "waterFlooding_well_location.inc"
    names: list[str] = []
    with path.open("w", newline="") as fh:
        w(fh, "GROUP 'ALL-WELLS' ATTACHTO 'FIELD'")
        for i in range(cfg.num_wells):
            row = cfg.locidx[int(loc_indices[i]) - 1]
            inject = selected_type(cfg, chromosome, i) or is_forced_injector(cfg, int(loc_indices[i]))
            name = well_name(int(row[0]), inject)
            names.append(name)
            w(fh, f"WELL  {name} ATTACHTO 'ALL-WELLS'")
        for i in range(cfg.num_wells):
            row = cfg.locidx[int(loc_indices[i]) - 1]
            inject = selected_type(cfg, chromosome, i) or is_forced_injector(cfg, int(loc_indices[i]))
            if inject:
                w(fh, f"INJECTOR MOBWEIGHT {names[i]}")
                w(fh, "INCOMP  WATER")
                w(fh, f"OPERATE  MAX  BHP   {cfg.injref}")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO {names[i]}")
                write_perf_layers(fh, int(row[1]), int(row[2]), 9)
            else:
                w(fh, f"PRODUCER {names[i]}")
                w(fh, f"OPERATE  MIN  BHP   {cfg.pref}")
                w(fh, "*MONITOR *WCUT   0.94   *SHUTIN")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO {names[i]}")
                write_perf_layers(fh, int(row[1]), int(row[2]), 8)
            w(fh, f"SHUTIN {names[i]}")
            w(fh, "")


def write_perf_layers(fh, i_grid: int, j_grid: int, last_layer: int) -> None:
    w(fh, f"{i_grid} {j_grid} 1  1.  OPEN FLOW-FROM 'SURFACE' REFLAYER")
    for layer in range(2, last_layer + 1):
        w(fh, f"{i_grid} {j_grid} {layer} 1.  OPEN FLOW-FROM {layer - 1}")


def write_channel_locations(cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray) -> None:
    """Write channelmodel locations and copy per-well PERF/LAYERXYZ blocks."""

    path = case_folder(cfg, fileid) / "waterFlooding_well_location.inc"
    names: list[str] = []
    with path.open("w", newline="") as fh:
        w(fh, "GROUP 'ALL-WELLS' ATTACHTO 'FIELD'")
        for i, loc in enumerate(loc_indices):
            inject = selected_type(cfg, chromosome, i)
            name = well_name(int(loc), inject)
            names.append(name)
            w(fh, f"WELL  {name} ATTACHTO 'ALL-WELLS'")
        for i, loc in enumerate(loc_indices):
            inject = selected_type(cfg, chromosome, i)
            well_data = cfg.source_dir / "modelpara" / f"W{int(loc):02d}.dat"
            if inject:
                w(fh, f"INJECTOR {names[i]}")
                w(fh, "INCOMP  WATER")
                w(fh, f"OPERATE  MAX  BHP   {cfg.injref}")
            else:
                w(fh, f"PRODUCER {names[i]}")
                w(fh, f"OPERATE  MIN  BHP   {cfg.pref}")
                w(fh, "OPERATE  MAX  STO   100000.0 CONT")
                w(fh, "MONITOR  MAX  WCUT  0.98     SHUTIN")
            copy_channel_perf(fh, well_data, names[i])
            w(fh, f"SHUTIN {names[i]}")


def copy_channel_perf(fh, well_data: Path, name: str) -> None:
    """Copy a channelmodel `W##.dat` perforation block under the new well name."""

    lines = well_data.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"Malformed well data file: {well_data}")
    idx = 2
    w(fh, f"PERF GEO {name}")
    w(fh, lines[idx])
    idx += 1
    while idx < len(lines) and "LAYERXYZ" not in lines[idx]:
        w(fh, lines[idx])
        idx += 1
    idx += 1
    w(fh, f"LAYERXYZ  {name}")
    if idx < len(lines):
        w(fh, lines[idx])
        idx += 1
    for line in lines[idx:]:
        w(fh, line)


def write_locations_order_only(cfg: CaseConfig, fileid: int) -> None:
    """Write fixed well locations for design_var=3 order-only optimization."""

    if cfg.locidx is None:
        raise RuntimeError("Order-only location writer needs locidx from baseinfo.mat/baseinfo1.mat.")
    if cfg.name == "channelmodel":
        write_channel_order_only_locations(cfg, fileid)
        return
    path = case_folder(cfg, fileid) / "waterFlooding_well_location.inc"
    names: list[str] = []
    active_rows = [(idx, row) for idx, row in enumerate(cfg.locidx, start=1) if row.shape[0] >= 4 and row[3] >= 0]
    with path.open("w", newline="") as fh:
        w(fh, "GROUP 'ALL-WELLS' ATTACHTO 'FIELD'")
        for row_idx, row in active_rows:
            inject = bool(row[3] == 1) or is_forced_injector(cfg, row_idx)
            name = well_name(int(row[0]), inject)
            names.append(name)
            w(fh, f"WELL  {name} ATTACHTO 'ALL-WELLS'")
        for (row_idx, row), name in zip(active_rows, names):
            inject = bool(row[3] == 1) or is_forced_injector(cfg, row_idx)
            if inject:
                w(fh, f"INJECTOR MOBWEIGHT {name}")
                w(fh, "INCOMP  WATER")
                w(fh, f"OPERATE  MAX  BHP   {cfg.injref}")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO {name}")
                write_perf_layers(fh, int(row[1]), int(row[2]), 9)
            else:
                w(fh, f"PRODUCER {name}")
                w(fh, f"OPERATE  MIN  BHP   {cfg.pref}")
                w(fh, "*MONITOR *WCUT   0.94   *SHUTIN")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO {name}")
                write_perf_layers(fh, int(row[1]), int(row[2]), 8)
            w(fh, f"SHUTIN {name}")
            w(fh, "")


def write_channel_order_only_locations(cfg: CaseConfig, fileid: int) -> None:
    """Write fixed channelmodel locations and types for order-only optimization."""

    locs, types = channel_order_only_locs_and_types(cfg)
    path = case_folder(cfg, fileid) / "waterFlooding_well_location.inc"
    names: list[str] = []
    with path.open("w", newline="") as fh:
        w(fh, "GROUP 'ALL-WELLS' ATTACHTO 'FIELD'")
        for loc, inject in zip(locs, types):
            name = well_name(int(loc), bool(inject))
            names.append(name)
            w(fh, f"WELL  {name} ATTACHTO 'ALL-WELLS'")
        for loc, inject, name in zip(locs, types, names):
            well_data = cfg.source_dir / "modelpara" / f"W{int(loc):02d}.dat"
            if inject:
                w(fh, f"INJECTOR {name}")
                w(fh, "INCOMP  WATER")
                w(fh, f"OPERATE  MAX  BHP   {cfg.injref}")
            else:
                w(fh, f"PRODUCER {name}")
                w(fh, f"OPERATE  MIN  BHP   {cfg.pref}")
                w(fh, "OPERATE  MAX  STO   100000.0 CONT")
                w(fh, "MONITOR  MAX  WCUT  0.98     SHUTIN")
            copy_channel_perf(fh, well_data, name)
            w(fh, f"SHUTIN {name}")


def write_schedule_order_type_location(
    cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray
) -> None:
    names = names_for_type_location(cfg, chromosome, loc_indices)
    path = case_folder(cfg, fileid) / "waterFlooding_sched.inc"
    with path.open("w", newline="") as fh:
        write_schedule_loop(cfg, fh, chromosome, names, loc_indices, use_order=True)


def write_schedule_type_location(cfg: CaseConfig, fileid: int, chromosome: np.ndarray, loc_indices: np.ndarray) -> None:
    names = names_for_type_location(cfg, chromosome, loc_indices)
    path = case_folder(cfg, fileid) / "waterFlooding_sched.inc"
    with path.open("w", newline="") as fh:
        write_schedule_loop(cfg, fh, chromosome, names, loc_indices, use_order=False)


def names_for_type_location(cfg: CaseConfig, chromosome: np.ndarray, loc_indices: np.ndarray) -> list[str]:
    names: list[str] = []
    for i, loc in enumerate(loc_indices):
        if cfg.name == "channelmodel":
            label = int(loc)
            inject = selected_type(cfg, chromosome, i)
        else:
            row = cfg.locidx[int(loc) - 1]
            label = int(row[0])
            inject = selected_type(cfg, chromosome, i) or is_forced_injector(cfg, int(loc))
        names.append(well_name(label, inject))
    return names


def write_schedule_loop(
    cfg: CaseConfig,
    fh,
    chromosome: np.ndarray,
    names: list[str],
    loc_indices: np.ndarray,
    use_order: bool,
) -> None:
    """Shared schedule writer for design_var 1 and 2.

    For design_var=1 the drilling order delays well activation by `Td`.
    For design_var=2 all wells are controlled from the first schedule step.
    """

    nc = 1
    nw = 1
    t = 0.1
    tpc = cfg.sim_time
    w(fh, f"TIME  {t}")
    while True:
        for j in range(cfg.num_wells):
            if (not use_order) or t >= (chromosome[j] - 1) * cfg.td:
                w(fh, "*TARGET    *BHP")
                w(fh, names[j])
                inject = selected_type(cfg, chromosome, j)
                if cfg.name != "channelmodel":
                    inject = inject or is_forced_injector(cfg, int(loc_indices[j]))
                w(fh, f"{cfg.injref if inject else cfg.pref:f}")
                w(fh, "")
        inner = 0
        while t < tpc * nc or (use_order and t < nw * cfg.td):
            t = t + 0.1 if inner == 0 else float(np.floor(t + 30 * inner))
            if use_order and t >= cfg.td * nw and nw < cfg.num_wells:
                w(fh, f"TIME  {cfg.td * nw}")
                nw += 1
                break
            if t >= tpc * nc:
                w(fh, f"TIME  {tpc * nc}")
                nc += 1
                break
            if t >= cfg.sim_time:
                w(fh, f"TIME  {cfg.sim_time}")
                break
            w(fh, f"TIME  {t}")
            inner = 1
        if t >= cfg.sim_time:
            break
        w(fh, "")


def write_schedule_order_only(cfg: CaseConfig, fileid: int, chromosome: np.ndarray) -> None:
    """Write the design_var=3 schedule using fixed locations and optimized order."""

    if cfg.locidx is None:
        raise RuntimeError("Order-only schedule writer needs locidx from baseinfo.mat/baseinfo1.mat.")
    if cfg.name == "channelmodel":
        write_channel_order_only_schedule(cfg, fileid, chromosome)
        return
    active_rows = [(idx, row) for idx, row in enumerate(cfg.locidx, start=1) if row.shape[0] >= 4 and row[3] >= 0]
    names = [
        well_name(int(row[0]), bool(row[3] == 1) or is_forced_injector(cfg, row_idx))
        for row_idx, row in active_rows
    ]
    loc_indices = np.arange(1, len(active_rows) + 1)
    path = case_folder(cfg, fileid) / "waterFlooding_sched.inc"
    with path.open("w", newline="") as fh:
        nc = 1
        nw = 1
        t = 0.1
        tpc = cfg.sim_time
        w(fh, f"TIME  {t}")
        while True:
            for j, (row_idx, row) in enumerate(active_rows):
                if t >= (chromosome[j] - 1) * cfg.td:
                    w(fh, "*TARGET    *BHP")
                    w(fh, names[j])
                    inject = bool(row[3] == 1) or is_forced_injector(cfg, row_idx)
                    w(fh, f"{cfg.injref if inject else cfg.pref:f}")
                    w(fh, "")
            inner = 0
            while t < tpc * nc or t < nw * cfg.td:
                t = t + 0.1 if inner == 0 else float(np.floor(t + 30 * inner))
                if t >= cfg.td * nw and nw < cfg.num_wells:
                    w(fh, f"TIME  {cfg.td * nw}")
                    nw += 1
                    break
                if t >= tpc * nc:
                    w(fh, f"TIME  {tpc * nc}")
                    nc += 1
                    break
                if t >= cfg.sim_time:
                    w(fh, f"TIME  {cfg.sim_time}")
                    break
                w(fh, f"TIME  {t}")
                inner = 1
            if t >= cfg.sim_time:
                break
            w(fh, "")


def write_channel_order_only_schedule(cfg: CaseConfig, fileid: int, chromosome: np.ndarray) -> None:
    """Write the channelmodel order-only schedule using fixed locations/types."""

    locs, types = channel_order_only_locs_and_types(cfg)
    names = [well_name(int(loc), bool(inject)) for loc, inject in zip(locs, types)]
    path = case_folder(cfg, fileid) / "waterFlooding_sched.inc"
    with path.open("w", newline="") as fh:
        nc = 1
        nw = 1
        t = 0.1
        tpc = cfg.sim_time
        w(fh, f"TIME  {t}")
        while True:
            for j, name in enumerate(names):
                if t >= (chromosome[j] - 1) * cfg.td:
                    w(fh, "*TARGET    *BHP")
                    w(fh, name)
                    w(fh, f"{cfg.injref if bool(types[j]) else cfg.pref:f}")
                    w(fh, "")
            inner = 0
            while t < tpc * nc or t < nw * cfg.td:
                t = t + 0.1 if inner == 0 else float(np.floor(t + 30 * inner))
                if t >= cfg.td * nw and nw < cfg.num_wells:
                    w(fh, f"TIME  {cfg.td * nw}")
                    nw += 1
                    break
                if t >= tpc * nc:
                    w(fh, f"TIME  {tpc * nc}")
                    nc += 1
                    break
                if t >= cfg.sim_time:
                    w(fh, f"TIME  {cfg.sim_time}")
                    break
                w(fh, f"TIME  {t}")
                inner = 1
            if t >= cfg.sim_time:
                break
            w(fh, "")


def channel_order_only_locs_and_types(cfg: CaseConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed channelmodel locations and types loaded from baseinfo."""

    if cfg.locidx is None:
        raise RuntimeError("Channel order-only optimization needs fixed locidx from baseinfo.")
    if cfg.well_type is None:
        raise RuntimeError("Channel order-only optimization needs fixed type from baseinfo.")
    locs = np.asarray(cfg.locidx, dtype=int).reshape(-1)[: cfg.num_wells]
    types = np.asarray(cfg.well_type, dtype=int).reshape(-1)[: cfg.num_wells]
    if locs.shape[0] < cfg.num_wells:
        raise RuntimeError(f"Channel order-only locidx has fewer than {cfg.num_wells} entries.")
    if types.shape[0] < cfg.num_wells:
        raise RuntimeError(f"Channel order-only type has fewer than {cfg.num_wells} entries.")
    return locs, types


def w(fh, line: str) -> None:
    """Write CMG include lines with Windows CRLF endings, like MATLAB `fprintf`."""

    fh.write(f"{line}\r\n")
