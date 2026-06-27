"""Objective evaluation and CMG simulator integration.

This is the Python version of `evalObjective.m`: generate simulator input
files, run each numbered CMG case folder, parse `waterFlooding.rwo`, and return
the minimized objective value `-NPV / scaling`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .encoding import decode_locations
from .writers import case_folder, write_case_inputs


@dataclass
class ObjectiveEvaluator:
    """Callable objective object used by the GA."""

    config: CaseConfig
    dry_run: bool = False
    retry_delay: float = 10.0
    stream_simulator_output: bool = False
    print_batch_timing: bool = True
    results_timeout_seconds: float | None = 60.0
    simulation_interrupt_timeout_seconds: float | None = 60.0
    evaluated: dict[tuple[int, ...], float] = field(default_factory=dict)
    count: int = 0
    cmg_batch_count: int = 0

    def __call__(self, population: np.ndarray) -> np.ndarray:
        """Evaluate one population, reusing cached chromosome results."""

        values = np.zeros(population.shape[0], dtype=float)
        pending: list[int] = []
        for k, chrom in enumerate(population):
            key = tuple(int(x) for x in chrom)
            if key in self.evaluated:
                values[k] = self.evaluated[key]
                self.count += 1
                print(f"{self.count} Simulation complete.", flush=True)
            else:
                pending.append(k)
        for batch_start in range(0, len(pending), self.config.num_parallel):
            # MATLAB evaluated simulations in parallel batches. This keeps the
            # same batching layout, using folders `1`, `2`, ... for each batch.
            batch = pending[batch_start : batch_start + self.config.num_parallel]
            batch_number = batch_start // self.config.num_parallel + 1
            if self.print_batch_timing:
                print(f"Starting simulation batch {batch_number} with {len(batch)} case(s).", flush=True)
            started = time.perf_counter()
            drilling_costs = []
            for local_id, pop_idx in enumerate(batch, start=1):
                chrom = population[pop_idx]
                locs = self.locations_for(chrom)
                write_case_inputs(self.config, local_id, chrom, locs)
                drilling_costs.append(self.drilling_cost(locs))
            if not self.dry_run:
                self.run_batch(len(batch))
            for local_id, pop_idx in enumerate(batch, start=1):
                chrom = population[pop_idx]
                if self.dry_run:
                    value = self.synthetic_objective(chrom)
                else:
                    value = self.value_from_rwo(local_id, drilling_costs[local_id - 1])
                values[pop_idx] = value
                self.evaluated[tuple(int(x) for x in chrom)] = value
                self.count += 1
                print(f"{self.count} Simulation complete.", flush=True)
            if self.print_batch_timing:
                elapsed = time.perf_counter() - started
                print(f"Batch {batch_number} complete in {elapsed:.1f} seconds.", flush=True)
        return values

    def locations_for(self, chrom: np.ndarray) -> np.ndarray:
        """Decode/lookup the selected locations for a chromosome."""

        if self.config.design_var in (1, 2):
            return decode_locations(
                chrom,
                self.config.beforeloc,
                self.config.num_wells,
                self.config.bits_per_location,
            )
        if self.config.locidx is not None and self.config.name == "channelmodel":
            return np.asarray(self.config.locidx).reshape(-1)[: self.config.num_wells]
        return np.arange(1, self.config.num_wells + 1)

    def drilling_cost(self, locs: np.ndarray) -> float:
        """Compute drilling cost for the case and selected locations."""

        if self.config.name != "channelmodel":
            return self.config.num_wells * self.config.cdrill_v
        if self.config.vertidx is None:
            return self.config.num_wells * self.config.cdrill_v
        total = 0.0
        for loc in locs[: self.config.num_wells]:
            idx = int(loc) - 1
            total += self.config.cdrill_v if bool(self.config.vertidx[idx]) else self.config.cdrill_h
        return total

    def run_batch(self, ncases: int) -> None:
        """Launch `RunCMG.bat` for each numbered case and retry on failure."""

        self.validate_batch_files(ncases)
        while True:
            self.cmg_batch_count += 1
            cmg_batch_id = self.cmg_batch_count
            if self.print_batch_timing:
                print(f"  CMG batch {cmg_batch_id}: starting {ncases} case(s).", flush=True)
            processes: list[tuple[int, Path, float, float, subprocess.Popen]] = []
            for local_id in range(1, ncases + 1):
                folder = case_folder(self.config, local_id)
                bat = folder / "RunCMG.bat"
                dat = folder / "waterFlooding.dat"
                clean_previous_case_outputs(folder)
                if self.print_batch_timing:
                    print(f"  CMG batch {cmg_batch_id}: launching case folder {local_id}: {folder}", flush=True)
                started = time.perf_counter()
                started_wall = time.time()
                process = subprocess.Popen(
                    [str(bat), str(dat)],
                    cwd=folder,
                    stdout=None if self.stream_simulator_output else subprocess.DEVNULL,
                    stderr=None if self.stream_simulator_output else subprocess.DEVNULL,
                    env=simulator_environment(self.config),
                )
                if self.print_batch_timing:
                    print(
                        f"  CMG batch {cmg_batch_id}: case folder {local_id} "
                        f"batch_pid={process.pid}",
                        flush=True,
                    )
                processes.append((local_id, folder, started, started_wall, process))
            for local_id, folder, started, started_wall, process in processes:
                return_code = self.wait_for_case_process(local_id, folder, started_wall, process)
                if self.print_batch_timing:
                    elapsed = time.perf_counter() - started
                    print(
                        f"  finished case folder {local_id}: {folder} "
                        f"(simulation wall time {elapsed:.1f} seconds, return code {return_code})",
                        flush=True,
                    )
            if self.batch_succeeded(ncases):
                return
            time.sleep(self.retry_delay)

    def wait_for_case_process(
        self,
        local_id: int,
        folder: Path,
        started: float,
        process: subprocess.Popen,
    ) -> int:
        """Wait for one case, timing out Results after IMEX finishes."""

        while True:
            return_code = process.poll()
            if return_code is not None:
                return int(return_code)
            if self.results_timeout_seconds is not None and self.results_report_timed_out(folder, started):
                if self.print_batch_timing:
                    print(
                        f"  case folder {local_id}: Results report timed out after "
                        f"{self.results_timeout_seconds:.0f} seconds; terminating batch process.",
                        flush=True,
                    )
                terminate_process_tree(process)
                return_code = process.wait()
                return int(return_code) if return_code is not None else -1
            if (
                self.simulation_interrupt_timeout_seconds is not None
                and self.simulation_interrupt_timed_out(folder, started)
            ):
                if self.print_batch_timing:
                    print(
                        f"  case folder {local_id}: IMEX is waiting at the Simulation Interrupt "
                        f"prompt after {self.simulation_interrupt_timeout_seconds:.0f} seconds; "
                        "terminating batch process.",
                        flush=True,
                    )
                terminate_process_tree(process)
                return_code = process.wait()
                return int(return_code) if return_code is not None else -1
            time.sleep(2.0)

    def results_report_timed_out(self, folder: Path, started: float) -> bool:
        """Return true when IMEX finished but Report.exe did not create RWO."""

        log = folder / "waterFlooding.log"
        rwo = folder / "waterFlooding.rwo"
        if not log.exists() or rwo.exists():
            return False
        if log.stat().st_mtime < started:
            return False
        text = log.read_text(errors="ignore")
        if "End of Simulation: Normal Termination" not in text:
            return False
        elapsed_since_log_update = time.time() - log.stat().st_mtime
        return elapsed_since_log_update >= float(self.results_timeout_seconds)

    def simulation_interrupt_timed_out(self, folder: Path, started: float) -> bool:
        """Return true when IMEX is waiting at its interactive interrupt menu."""

        log = folder / "waterFlooding.log"
        if not log.exists():
            return False
        if log.stat().st_mtime < started:
            return False
        text = log.read_text(errors="ignore")
        if "End of Simulation: Normal Termination" in text:
            return False
        if "Simulation Interrupt" not in text or "==> Enter Choice:" not in text:
            return False
        elapsed_since_log_update = time.time() - log.stat().st_mtime
        return elapsed_since_log_update >= float(self.simulation_interrupt_timeout_seconds)

    def validate_batch_files(self, ncases: int) -> None:
        """Fail early when a batch file points to missing simulator programs."""

        missing: list[str] = []
        for local_id in range(1, ncases + 1):
            folder = case_folder(self.config, local_id)
            bat = folder / "RunCMG.bat"
            if not bat.exists():
                missing.append(str(bat))
                continue
            text = bat.read_text(errors="ignore")
            for exe in re.findall(r'"([A-Za-z]:\\[^"]+\.exe)"', text, flags=re.IGNORECASE):
                if not Path(exe).exists():
                    missing.append(exe)
        if missing:
            unique = "\n  - ".join(dict.fromkeys(missing))
            raise FileNotFoundError(
                "CMG could not be launched because these paths do not exist:\n"
                f"  - {unique}\n"
                "Update the CMG executable paths in the case template RunCMG.bat, "
                "then recreate or update the numbered work folders."
            )

    def batch_succeeded(self, ncases: int) -> bool:
        """Check CMG logs and required `waterFlooding.rwo` files."""

        for local_id in range(1, ncases + 1):
            folder = case_folder(self.config, local_id)
            ok, reason = case_succeeded(folder)
            if not ok:
                if self.print_batch_timing:
                    print(f"  case folder {local_id}: retrying batch because {reason}.", flush=True)
                return False
            log = folder / "waterFlooding.log"
            try:
                log.unlink()
            except OSError:
                pass
        return True

    def value_from_rwo(self, local_id: int, drilling_cost: float) -> float:
        """Parse one simulator output file into the minimized objective."""

        path = case_folder(self.config, local_id) / "waterFlooding.rwo"
        try:
            obj = parse_npv(path, self.config, drilling_cost)
            try:
                path.unlink()
            except OSError:
                pass
            return -obj / self.config.objective_scaling
        except Exception:
            return 1000.0

    def synthetic_objective(self, chrom: np.ndarray) -> float:
        # Useful for fast GA smoke tests when CMG is unavailable.
        order_score = np.sum(chrom[: self.config.num_wells] * np.arange(1, self.config.num_wells + 1))
        return float((np.sum(chrom[self.config.num_wells :]) + 0.1 * order_score) / 1000.0)


def parse_npv(path: Path, cfg: CaseConfig, drilling_cost: float) -> float:
    """Read CMG cumulative rates and calculate discounted NPV."""

    lines = path.read_text(errors="ignore").splitlines()[9:]
    values: list[float] = []
    for line in lines:
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                continue
    if len(values) < 6:
        raise RuntimeError(f"No numeric RWO data found in {path}")
    usable = len(values) - (len(values) % 6)
    qt = np.asarray(values[:usable], dtype=float).reshape(-1, 6)
    qprior = np.zeros(6)
    obj = 0.0
    for row in qt:
        # CMG output is cumulative, so each row is differenced from the prior
        # row before applying prices and discounting.
        q = row - qprior
        t = row[0]
        discount = (1.0 + cfg.npv.discount_factor) ** (-t / 365.0)
        obj += discount * (
            cfg.npv.oil_price * q[1]
            - cfg.npv.water_production_cost * q[3]
            - cfg.npv.water_production_cost * q[4]
        )
        qprior = row
    if cfg.name == "channelmodel":
        return obj - drilling_cost
    return obj * 6.289814 - drilling_cost


def clean_previous_case_outputs(folder: Path) -> None:
    """Remove generated CMG outputs so a new launch cannot read stale files."""

    for name in (
        "waterFlooding.log",
        "waterFlooding.rwo",
        "waterFlooding.sr3",
        "waterFlooding.irf",
    ):
        path = folder / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def case_succeeded(folder: Path) -> tuple[bool, str]:
    """Return whether a simulator folder has valid outputs for objective parsing."""

    log = folder / "waterFlooding.log"
    rwo = folder / "waterFlooding.rwo"
    sr3 = folder / "waterFlooding.sr3"
    if not log.exists():
        return False, "waterFlooding.log is missing"
    if not rwo.exists():
        return False, "waterFlooding.rwo is missing"
    if not sr3.exists():
        return False, "waterFlooding.sr3 is missing"

    log_text = log.read_text(errors="ignore")
    rwo_text = rwo.read_text(errors="ignore")
    combined = f"{log_text}\n{rwo_text}"
    failure_markers = (
        "End of Simulation: Abnormal Termination",
        "FATAL ERROR",
        "LICENSING",
        "Cannot connect to server",
        "Invalid File",
        "No Table Data",
    )
    for marker in failure_markers:
        if marker in combined:
            return False, f"output contains '{marker}'"
    if "Simulation Interrupt" in log_text and "==> Enter Choice:" in log_text:
        return False, "simulator is waiting at the Simulation Interrupt prompt"
    if "End of Simulation: Normal Termination" not in log_text:
        return False, "normal simulator termination was not confirmed"
    if log_text.count("Error messages") > 2:
        return False, "simulator log reports error messages"
    return True, "outputs are complete"


def simulator_environment(cfg: CaseConfig) -> dict[str, str] | None:
    """Return environment variables for one simulator process."""

    env = os.environ.copy()
    add_windows_environment_variable(env, "CMG_LIC_HOST")
    add_windows_environment_variable(env, "RLM_LICENSE")
    add_windows_environment_variable(env, "LM_LICENSE_FILE")
    if cfg.simulation_threads is None:
        return env
    threads = str(int(cfg.simulation_threads))
    # These are honored by many numerical libraries and some simulator builds.
    # If a CMG version needs a command-line thread flag, add it in RunCMG.bat.
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["CMG_NUM_THREADS"] = threads
    return env


def add_windows_environment_variable(env: dict[str, str], name: str) -> None:
    """Copy user/machine Windows env vars when the current process lacks them."""

    if os.name != "nt" or env.get(name):
        return
    try:
        import winreg
    except ImportError:
        return
    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for hive, key_name in locations:
        try:
            with winreg.OpenKey(hive, key_name) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            env[name] = str(value)
            return


def terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a Windows batch process and its child simulator/report process."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()


def prepare_work_folders(cfg: CaseConfig) -> None:
    """Ensure numbered simulator folders exist in the Python work directory."""

    if cfg.template_dir is None or not cfg.template_dir.exists():
        raise FileNotFoundError(
            "TEMPLATE_DIR must point to a reusable simulator template folder. "
            f"Current value: {cfg.template_dir}"
        )

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, cfg.num_parallel + 1):
        dst = cfg.work_dir / str(i)
        if dst.exists():
            continue
        shutil.copytree(cfg.template_dir, dst)


def clean_generated_work_folders(cfg: CaseConfig, clean_history: bool = True) -> None:
    """Remove generated simulator folders from a previous optimization run.

    Only numeric folders such as `work/1`, `work/2`, ... are removed. Case
    inputs in the work root, such as `baseinfo_locidx.csv` and `inj_idx.csv`,
    are intentionally left alone.
    """

    if not cfg.work_dir.exists():
        return
    removed: list[Path] = []
    for child in cfg.work_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            shutil.rmtree(child)
            removed.append(child)
    if clean_history:
        history = cfg.work_dir / "python_tempdata"
        if history.exists() and history.is_dir():
            shutil.rmtree(history)
            removed.append(history)
    if removed:
        print(f"Cleaned {len(removed)} generated work item(s) from {cfg.work_dir}.", flush=True)
