"""Simulator input writers translated from chapter 4 `writeloc.m`/`writesch.m`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CaseConfig
from .variables import is_oil_cell, well_slice


def case_folder(cfg: CaseConfig, fileid: int) -> Path:
    return Path(cfg.work_dir) / str(fileid)


def write_case_inputs(cfg: CaseConfig, fileid: int, u: np.ndarray) -> None:
    """Write simulator inputs for one combined `[active,type,i,j]` vector."""

    if cfg.name == "Brugge":
        write_brugge_locations(cfg, fileid, u)
        write_brugge_schedule(cfg, fileid, u)
    elif cfg.name == "PUNQ":
        write_punq_schedule(cfg, fileid, u)
    else:
        raise ValueError(f"Unsupported case: {cfg.name}")


def well_kind(cfg: CaseConfig, u: np.ndarray, widx: int) -> str:
    """Return `producer` only when active, type=0, and the grid cell is oil."""

    sl = well_slice(cfg, widx)
    is_type_producer = int(round(u[sl.start + 1])) == 0
    return "producer" if is_type_producer and is_oil_cell(cfg, int(u[sl.start + 2]), int(u[sl.start + 3])) else "injector"


def well_label(cfg: CaseConfig, widx: int) -> int:
    if cfg.locidx is not None and widx < cfg.locidx.shape[0]:
        return int(cfg.locidx[widx, 0])
    return widx + 1


def write_brugge_locations(cfg: CaseConfig, fileid: int, u: np.ndarray) -> None:
    """Write `waterFlooding_well_location.inc` for the Brugge CMG case."""

    path = case_folder(cfg, fileid) / "waterFlooding_well_location.inc"
    with path.open("w", newline="") as fh:
        w(fh, "GROUP 'Group'    ''   ATTACHTO 'Field'")
        for idx in range(cfg.num_wells):
            sl = well_slice(cfg, idx)
            if int(round(u[sl.start])) != 1:
                continue
            label = well_label(cfg, idx)
            prefix = "P" if well_kind(cfg, u, idx) == "producer" else "I"
            w(fh, f"WELL  '{prefix}{label}' ATTACHTO 'Group'")
        w(fh, "")
        for idx in range(cfg.num_wells):
            sl = well_slice(cfg, idx)
            if int(round(u[sl.start])) != 1:
                continue
            label = well_label(cfg, idx)
            i_grid = int(u[sl.start + 2])
            j_grid = int(u[sl.start + 3])
            if well_kind(cfg, u, idx) == "producer":
                w(fh, f"PRODUCER 'P{label}'")
                w(fh, "OPERATE  MIN BHP 50.973")
                w(fh, "*MONITOR *WCUT   0.94   *SHUTIN")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO 'P{label}'")
                write_perf_layers(fh, i_grid, j_grid, 8)
                w(fh, f"SHUTIN 'P{label}'")
            else:
                w(fh, f"INJECTOR MOBWEIGHT 'I{label}'")
                w(fh, "INCOMP  WATER")
                w(fh, "OPERATE MAX BHP 183.572")
                w(fh, "GEOMETRY  K  0.0762  0.37  1.  0.")
                w(fh, f"PERF GEO 'I{label}'")
                write_perf_layers(fh, i_grid, j_grid, 9)
                w(fh, f"SHUTIN 'I{label}'")
                w(fh, "")


def write_perf_layers(fh, i_grid: int, j_grid: int, last_layer: int) -> None:
    w(fh, f"{i_grid} {j_grid} 1  1.  OPEN FLOW-FROM 'SURFACE' REFLAYER")
    for layer in range(2, last_layer + 1):
        w(fh, f"{i_grid} {j_grid} {layer} 1.  OPEN FLOW-FROM {layer - 1}")


def write_brugge_schedule(cfg: CaseConfig, fileid: int, u: np.ndarray) -> None:
    """Write `waterFlooding_sched.inc` for Brugge."""

    path = case_folder(cfg, fileid) / "waterFlooding_sched.inc"
    with path.open("w", newline="") as fh:
        t = 0.1
        nc = 1
        w(fh, f"TIME  {t}")
        for idx in range(cfg.num_wells):
            sl = well_slice(cfg, idx)
            if int(round(u[sl.start])) != 1:
                continue
            label = well_label(cfg, idx)
            if well_kind(cfg, u, idx) == "producer":
                w(fh, "*TARGET    *BHP")
                w(fh, f'"P{label}"')
                w(fh, "50.973")
                w(fh, "")
            else:
                w(fh, "*TARGET    *BHP")
                w(fh, f'"I{label}"')
                w(fh, "183.5715")
                w(fh, "")
        while True:
            inner = 0
            while t < cfg.sim_time * nc:
                t = t + 0.1 if inner == 0 else float(np.floor(t + 30 * inner))
                if t >= cfg.sim_time * nc:
                    w(fh, f"TIME  {cfg.sim_time * nc}")
                    nc += 1
                    break
                w(fh, f"TIME  {t}")
                inner = 1
            if t >= cfg.sim_time:
                break
            w(fh, "")


def write_punq_schedule(cfg: CaseConfig, fileid: int, u: np.ndarray) -> None:
    """Write Eclipse `PUN_SCH.inc` for PUNQ."""

    path = case_folder(cfg, fileid) / "PUN_SCH.inc"
    with path.open("w", newline="") as fh:
        fh.write("ECHO \n\nRPTSCHED \n")
        fh.write("'PRES' 'SOIL' 'RESTART=2' / \n\nRPTRST \n")
        fh.write("'BASIC=2' / \n\n")
        for idx in range(cfg.num_wells):
            sl = well_slice(cfg, idx)
            if int(round(u[sl.start])) != 1:
                continue
            i_grid = int(u[sl.start + 2])
            j_grid = int(u[sl.start + 3])
            if well_kind(cfg, u, idx) == "producer":
                fh.write("WELSPECS\n")
                fh.write(f"'PRO-{idx + 1}' 'G1' {i_grid} {j_grid} 1* 'OIL' 3* 'NO' 5* 'STD' /\n/\n")
                fh.write("COMPDAT\n")
                fh.write(f"'PRO-{idx + 1}' 2* 1 5 'OPEN' 2* 0.1 1* 0 3* / \n/\n")
                fh.write("WCONPROD\n")
                fh.write(f"'PRO-{idx + 1}' 'OPEN' 'BHP' 5* 120  3* / \n/\n")
                fh.write("WCUTBACK\n")
                fh.write(f"'PRO-{idx + 1}' 1* 200 2* 0.75 'OIL' 6* /\n")
                fh.write("/\n-------------------------------------------------------\n")
            else:
                fh.write("WELSPECS\n")
                fh.write(f"'INJ-{idx + 1}' 'G2' {i_grid} {j_grid} 1* 'WATER' 3* 'NO' 5* 'STD' /\n/\n")
                fh.write("COMPDAT\n")
                fh.write(f"'INJ-{idx + 1}' 2* 1 5 'OPEN' 2* 0.1 1* 0 3* / \n/\n")
                fh.write("WCONINJE\n")
                fh.write(f"'INJ-{idx + 1}' 'WATER' 'OPEN' 'RATE' 50   1* 351.69   4* /\n\n/\n")
        fh.write("\nTSTEP \n10 / \n\nTSTEP \n243*30 / \n\nEND")


def w(fh, line: str) -> None:
    fh.write(f"{line}\r\n")
