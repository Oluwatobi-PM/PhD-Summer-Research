"""Objective functions and simulator execution for chapter 4."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .variables import active_well_count, well_slice
from .writers import case_folder, write_case_inputs


@dataclass
class ObjectiveEvaluator:
    """Callable objective object used by all GA variants."""

    config: CaseConfig
    dry_run: bool = False
    retry_delay: float = 10.0
    evaluated: dict[tuple[int, ...], float] = field(default_factory=dict)
    count: int = 0

    def __call__(self, population: np.ndarray) -> np.ndarray:
        population = np.asarray(population, dtype=float)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        values = np.zeros(population.shape[0], dtype=float)
        pending: list[int] = []
        for idx, chrom in enumerate(population):
            key = tuple(int(round(x)) for x in chrom)
            if key in self.evaluated:
                values[idx] = self.evaluated[key]
                self.count += 1
                print(f"{self.count} Simulation complete.")
            else:
                pending.append(idx)
        for batch_start in range(0, len(pending), self.config.num_parallel):
            batch = pending[batch_start : batch_start + self.config.num_parallel]
            nwell = []
            for local_id, pop_idx in enumerate(batch, start=1):
                chrom = population[pop_idx]
                write_case_inputs(self.config, local_id, chrom)
                nwell.append(active_well_count(self.config, chrom))
            if not self.dry_run:
                self.run_batch(len(batch))
            for local_id, pop_idx in enumerate(batch, start=1):
                chrom = population[pop_idx]
                if self.dry_run:
                    value = self.synthetic_objective(chrom)
                else:
                    value = self.value_from_output(local_id, nwell[local_id - 1])
                values[pop_idx] = value
                self.evaluated[tuple(int(round(x)) for x in chrom)] = value
                self.count += 1
                print(f"{self.count} Simulation complete.")
        return values

    def run_batch(self, ncases: int) -> None:
        """Launch each numbered simulator case and retry until outputs exist."""

        while True:
            for local_id in range(1, ncases + 1):
                folder = case_folder(self.config, local_id)
                if self.config.name == "Brugge":
                    bat = folder / "RunCMG.bat"
                    dat = folder / "waterFlooding.dat"
                    subprocess.run([str(bat), str(dat)], cwd=folder, stdout=subprocess.DEVNULL, check=False)
                else:
                    bat = folder / "RunEclipse.bat"
                    subprocess.run([str(bat)], cwd=folder, stdout=subprocess.DEVNULL, check=False)
            if self.batch_succeeded(ncases):
                return
            time.sleep(self.retry_delay)

    def batch_succeeded(self, ncases: int) -> bool:
        for local_id in range(1, ncases + 1):
            folder = case_folder(self.config, local_id)
            if self.config.name == "Brugge":
                log = folder / "waterFlooding.log"
                output = folder / "waterFlooding.rwo"
                if not log.exists() or not output.exists():
                    return False
                if log.read_text(errors="ignore").count("Error messages") > 2:
                    return False
                try:
                    log.unlink()
                except OSError:
                    pass
            else:
                prt = folder / "PUN_E100.PRT"
                output = folder / "PUN_E100.RSM"
                if not prt.exists() or not output.exists():
                    return False
                if "Error summary" in prt.read_text(errors="ignore"):
                    return False
                try:
                    prt.unlink()
                except OSError:
                    pass
        return True

    def value_from_output(self, local_id: int, nwell: int) -> float:
        try:
            folder = case_folder(self.config, local_id)
            if self.config.name == "Brugge":
                obj = parse_brugge_rwo(folder / "waterFlooding.rwo", self.config, nwell)
                (folder / "waterFlooding.rwo").unlink(missing_ok=True)
            else:
                obj = parse_punq_rsm(folder / "PUN_E100.RSM", self.config, nwell)
                (folder / "PUN_E100.RSM").unlink(missing_ok=True)
            return -obj / self.config.objective_scaling
        except Exception:
            return 1000.0

    def synthetic_objective(self, chrom: np.ndarray) -> float:
        """Fast deterministic score for dry smoke tests, not research output."""

        total = 0.0
        for idx in range(self.config.num_wells):
            sl = well_slice(self.config, idx)
            if int(round(chrom[sl.start])) == 1:
                total += chrom[sl.start + 2] ** 2 + chrom[sl.start + 3] ** 2
                total += 10.0 * chrom[sl.start + 1]
        return float(total / 100000.0)


def parse_brugge_rwo(path: Path, cfg: CaseConfig, nwell: int) -> float:
    values = numeric_payload(path, skip_lines=9)
    qt = values[: len(values) - len(values) % 6].reshape(-1, 6)
    obj = 0.0
    prior = np.zeros(6)
    for row in qt:
        q = row - prior
        t = row[0]
        discount = (1.0 + cfg.npv.discount_factor) ** (-t / 365.0)
        obj += discount * (cfg.npv.oil_price * q[1] - cfg.npv.water_production_cost * q[3] - cfg.npv.water_production_cost * q[4])
        prior = row
    return obj * 6.289814 - nwell * cfg.cdrill


def parse_punq_rsm(path: Path, cfg: CaseConfig, nwell: int) -> float:
    lines = path.read_text(errors="ignore").splitlines()
    unit_line = lines[4] if len(lines) >= 5 else ""
    values = []
    for line in lines[7:]:
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                continue
    arr = np.asarray(values, dtype=float)
    qt = arr[: len(arr) - len(arr) % 5].reshape(-1, 5)
    units = unit_line.split()
    for col in range(min(5, len(units) - 1)):
        multiplier = unit_multiplier(units[col + 1])
        qt[:, col] *= multiplier
    obj = 0.0
    prior = np.zeros(5)
    for row in qt:
        q = row - prior
        t = row[0]
        discount = (1.0 + cfg.npv.discount_factor) ** (-t / 365.0)
        obj += discount * (cfg.npv.oil_price * q[2] - cfg.npv.water_production_cost * q[3] - cfg.npv.water_production_cost * q[4])
        prior = row
    return obj * 6.289814 - nwell * cfg.cdrill


def numeric_payload(path: Path, skip_lines: int) -> np.ndarray:
    values: list[float] = []
    for line in path.read_text(errors="ignore").splitlines()[skip_lines:]:
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                continue
    if not values:
        raise RuntimeError(f"No numeric data found in {path}")
    return np.asarray(values, dtype=float)


def unit_multiplier(unit: str) -> float:
    """Interpret simple Eclipse unit strings such as `*10**3`."""

    unit = unit.strip().replace("**", "^")
    if not unit:
        return 1.0
    if unit.startswith("*10^"):
        try:
            return 10.0 ** float(unit.split("^", 1)[1])
        except ValueError:
            return 1.0
    return 1.0


def prepare_work_folders(cfg: CaseConfig) -> None:
    """Copy numbered simulator folders into the Python work directory."""

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, cfg.num_parallel + 1):
        src = cfg.source_dir / str(i)
        dst = cfg.work_dir / str(i)
        if dst.exists():
            continue
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
