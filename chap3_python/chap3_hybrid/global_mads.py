"""Hybrid global optimizer plus NOMAD/MADS local refinement.

The default PSO coupling follows the Isebor-style handoff: advance PSO while
it improves, launch MADS from the current best particle after a non-improving
PSO iteration, keep MADS going only while it improves, then reduce the MADS
frame size and return to PSO after a non-improving MADS episode.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import numpy as np

from chap3_ga.checkpoint import atomic_savez, best_population_payload, common_metadata, optimizer_history_dir
from chap3_ga.config import CaseConfig
from chap3_ga.encoding import decode_locations
from chap3_ga.lhs_initialization import decode_lhs_population, lhs_population, normalized_dimension
from chap3_ga.objective import DesignPopulationEvaluator, ObjectiveEvaluator
from chap3_mads.mads import MADSData, MADSVariableSpace, design_cache_key, make_block_callback, nomad_parameters
from chap3_pso.pso import PSOData, evaluate_swarm, initial_particles, initial_velocities, mutate_particles


@dataclass
class HybridMADSData:
    """Mutable state for a global-search + MADS hybrid run."""

    config: CaseConfig
    objective: ObjectiveEvaluator
    global_optimizer: str = "pso"
    max_simulations: int | None = None
    pso_handoff_rule: str = "no_improvement"
    pso_stall_iterations: int = 1
    global_iterations_per_cycle: int = 2
    local_mads_budget: int = 100
    mads_iterations_per_episode: int = 1
    mads_frame_reduction: float = 0.5
    initial_poll_size: float = 1.0
    min_poll_size: float = 1.0
    direction_type: str = "ORTHO 2N"
    display_degree: int = 0
    bb_max_block_size: int | None = None
    seed: int = 1000
    omega: float = 0.7298
    phip: float = 1.496
    phig: float = 1.496
    velocity_clamp: float | None = None
    mutation_rate: float = 0.0
    improvement_tolerance: float = 0.0
    initialization: str = "lhs"
    initialization_seed: int | None = None
    initial_velocity: str = "zero"
    pso: PSOData | None = None
    mads_cache: dict[tuple, float] = field(default_factory=dict)
    cycle: int = 0
    pso_iteration: int = 0
    mads_frame_size: float = 1.0
    best_chromosome: np.ndarray | None = None
    xmin: np.ndarray | None = None
    best_particle: np.ndarray | None = None
    best_objective: float = np.inf
    phase_history: list[str] = field(default_factory=list)
    eval_history: list[int] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    best_chrom_history: list[np.ndarray] = field(default_factory=list)
    mads_frame_history: list[float] = field(default_factory=list)
    diagnostic_rows: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.global_optimizer = str(self.global_optimizer).strip().lower()
        if self.global_optimizer != "pso":
            raise ValueError("Hybrid MADS currently supports GLOBAL_OPTIMIZER='pso'.")
        self.pso_handoff_rule = str(self.pso_handoff_rule).strip().lower()
        if self.pso_handoff_rule not in ("no_improvement", "fixed"):
            raise ValueError("PSO_HANDOFF_RULE must be 'no_improvement' or 'fixed'.")
        self.max_simulations = self.config.maxgen if self.max_simulations is None else int(self.max_simulations)
        if int(self.max_simulations) <= 0:
            raise ValueError("max_simulations must be positive.")
        if int(self.local_mads_budget) <= 0:
            raise ValueError("local_mads_budget must be positive.")
        if int(self.global_iterations_per_cycle) <= 0:
            raise ValueError("global_iterations_per_cycle must be positive.")
        if int(self.pso_stall_iterations) <= 0:
            raise ValueError("pso_stall_iterations must be positive.")
        if int(self.mads_iterations_per_episode) <= 0:
            raise ValueError("mads_iterations_per_episode must be positive.")
        if not 0.0 < float(self.mads_frame_reduction) <= 1.0:
            raise ValueError("mads_frame_reduction must be in (0, 1].")
        self.mads_frame_size = max(float(self.initial_poll_size), float(self.min_poll_size), 1.0)
        self.bb_max_block_size = self.config.num_parallel if self.bb_max_block_size is None else int(self.bb_max_block_size)


def run_hybrid_mads(hybrid: HybridMADSData, save_history: bool = True) -> HybridMADSData:
    """Run PSO-MADS with improvement-based MADS handoffs."""

    try:
        import PyNomad
    except ImportError as exc:
        raise ImportError("PyNomadBBO is required for hybrid MADS. Install it with `pip install PyNomadBBO`.") from exc

    rng = np.random.default_rng(int(hybrid.seed))
    PyNomad.setSeed(int(hybrid.seed))
    initialize_pso_state(hybrid, rng)
    if int(hybrid.max_simulations) < int(hybrid.pso.swarm_size):
        raise ValueError("MAX_SIMULATIONS must be at least SWARM_SIZE for the initial PSO evaluation.")

    before_initial_simulated = int(hybrid.objective.simulated_count)
    before_initial_cache_hits = int(hybrid.objective.cache_hits)
    evaluate_swarm(
        hybrid.pso,
        hybrid.pso.particles,
        hybrid.pso.velocities,
        hybrid.pso.personal_best_particles,
        hybrid.pso.personal_best_obj,
        iteration=0,
        save_history=False,
    )
    sync_from_pso(hybrid)
    record_diagnostic_row(
        hybrid,
        phase="pso_initial",
        source="pso",
        best_before=np.nan,
        best_after=float(hybrid.best_objective),
        simulated_before=before_initial_simulated,
        simulated_after=int(hybrid.objective.simulated_count),
        cache_hits_before=before_initial_cache_hits,
        cache_hits_after=int(hybrid.objective.cache_hits),
        callback_evaluations=np.nan,
        unique_designs=np.nan,
        pso_mean_npv=mean_npv_from_objectives(hybrid.pso.objv),
        pso_median_npv=median_npv_from_objectives(hybrid.pso.objv),
        pso_current_best_npv=current_best_npv_from_objectives(hybrid.pso.objv),
        failed_evaluations=count_failed_objectives(hybrid.pso.objv),
        mads_frame_before=float(hybrid.mads_frame_size),
        mads_frame_after=float(hybrid.mads_frame_size),
        improved=True,
    )
    record_hybrid_state(hybrid, "pso_initial", save_history)

    while remaining_budget(hybrid) > 0:
        hybrid.cycle += 1
        print(f"Hybrid cycle {hybrid.cycle} started.", flush=True)
        ran_pso, pso_stalled = run_pso_until_handoff(hybrid, rng, save_history)
        if remaining_budget(hybrid) <= 0:
            break
        if hybrid.pso.best_particle is None:
            break
        if not pso_stalled and hybrid.pso_handoff_rule == "no_improvement":
            break
        improved_by_mads = run_mads_improvement_phase(hybrid, PyNomad, save_history)
        if improved_by_mads:
            inject_mads_best_into_pso(hybrid)
            record_hybrid_state(hybrid, "mads_improved", save_history)
        else:
            record_hybrid_state(hybrid, "mads_failed", save_history)
        if not ran_pso and not improved_by_mads:
            break

    save_hybrid_data(hybrid)
    print_results(hybrid)
    return hybrid


def initialize_pso_state(hybrid: HybridMADSData, rng: np.random.Generator) -> None:
    """Create the PSO state used by the hybrid global phase."""

    cfg = hybrid.config
    initial = None
    mode = str(hybrid.initialization).strip().lower()
    if mode == "lhs":
        init_seed = int(hybrid.initialization_seed if hybrid.initialization_seed is not None else hybrid.seed)
        initial, _ = lhs_population(cfg.population_size, normalized_dimension(cfg), np.random.default_rng(init_seed))
    elif mode != "random":
        raise ValueError(f"Unsupported INITIALIZATION={hybrid.initialization!r}. Expected 'lhs' or 'random'.")

    pso = PSOData(
        cfg,
        hybrid.objective,
        max_iterations=1,
        swarm_size=cfg.population_size,
        omega=hybrid.omega,
        phip=hybrid.phip,
        phig=hybrid.phig,
        velocity_clamp=hybrid.velocity_clamp,
        mutation_rate=hybrid.mutation_rate,
        improvement_tolerance=hybrid.improvement_tolerance,
        initial_particles=initial,
        initial_velocity=hybrid.initial_velocity,
    )
    pso.particles = initial_particles(pso, rng)
    pso.velocities = initial_velocities(pso, rng)
    pso.personal_best_particles = pso.particles.copy()
    pso.personal_best_obj = np.full(int(pso.swarm_size), np.inf, dtype=float)
    hybrid.pso = pso


def run_pso_until_handoff(
    hybrid: HybridMADSData,
    rng: np.random.Generator,
    save_history: bool,
) -> tuple[bool, bool]:
    """Advance PSO until the configured handoff condition is met."""

    pso = require_pso(hybrid)
    ran = False
    stalled = 0
    max_iterations = (
        int(hybrid.global_iterations_per_cycle)
        if hybrid.pso_handoff_rule == "fixed"
        else max(1, remaining_budget(hybrid) // int(pso.swarm_size))
    )
    for _ in range(max_iterations):
        if remaining_budget(hybrid) < int(pso.swarm_size):
            print("Skipping PSO block because remaining budget is smaller than the swarm size.", flush=True)
            return ran, False
        improved = run_one_pso_iteration(hybrid, rng, save_history)
        ran = True
        if hybrid.pso_handoff_rule == "fixed":
            continue
        if improved:
            stalled = 0
            continue
        stalled += 1
        if stalled >= int(hybrid.pso_stall_iterations):
            print("PSO returned control to MADS after a non-improving iteration.", flush=True)
            return ran, True
    return ran, hybrid.pso_handoff_rule == "fixed"


def run_one_pso_iteration(hybrid: HybridMADSData, rng: np.random.Generator, save_history: bool) -> bool:
    """Advance PSO by one iteration and return whether the global best improved."""

    pso = require_pso(hybrid)
    previous_best = float(pso.fxmin)
    before_simulated = int(hybrid.objective.simulated_count)
    before_cache_hits = int(hybrid.objective.cache_hits)
    hybrid.pso_iteration += 1
    rp = rng.uniform(size=pso.particles.shape)
    rg = rng.uniform(size=pso.particles.shape)
    pso.velocities = (
        float(pso.omega) * pso.velocities
        + float(pso.phip) * rp * (pso.personal_best_particles - pso.particles)
        + float(pso.phig) * rg * (pso.best_particle - pso.particles)
    )
    if pso.velocity_clamp is not None:
        pso.velocities = np.clip(pso.velocities, -float(pso.velocity_clamp), float(pso.velocity_clamp))
    pso.particles = np.clip(pso.particles + pso.velocities, 0.0, 1.0)
    mutate_particles(pso.particles, float(pso.mutation_rate), rng)
    evaluate_swarm(
        pso,
        pso.particles,
        pso.velocities,
        pso.personal_best_particles,
        pso.personal_best_obj,
        iteration=hybrid.pso_iteration,
        save_history=False,
    )
    sync_from_pso(hybrid)
    improved = pso.fxmin < previous_best - float(hybrid.improvement_tolerance)
    record_diagnostic_row(
        hybrid,
        phase="pso_improved" if improved else "pso_no_improvement",
        source="pso",
        best_before=previous_best,
        best_after=float(pso.fxmin),
        simulated_before=before_simulated,
        simulated_after=int(hybrid.objective.simulated_count),
        cache_hits_before=before_cache_hits,
        cache_hits_after=int(hybrid.objective.cache_hits),
        callback_evaluations=np.nan,
        unique_designs=np.nan,
        pso_mean_npv=mean_npv_from_objectives(pso.objv),
        pso_median_npv=median_npv_from_objectives(pso.objv),
        pso_current_best_npv=current_best_npv_from_objectives(pso.objv),
        failed_evaluations=count_failed_objectives(pso.objv),
        mads_frame_before=float(hybrid.mads_frame_size),
        mads_frame_after=float(hybrid.mads_frame_size),
        improved=improved,
    )
    record_hybrid_state(hybrid, "pso_improved" if improved else "pso_no_improvement", save_history)
    return improved


def run_mads_improvement_phase(hybrid: HybridMADSData, py_nomad, save_history: bool) -> bool:
    """Run MADS episodes until one episode fails to improve."""

    pso = require_pso(hybrid)
    if pso.xmin is None:
        return False

    design_objective = DesignPopulationEvaluator(hybrid.objective)
    space = MADSVariableSpace(hybrid.config)
    mads = MADSData(
        hybrid.config,
        design_objective,
        max_simulations=min(int(hybrid.local_mads_budget), remaining_budget(hybrid)),
        initial_poll_size=hybrid.mads_frame_size,
        min_poll_size=hybrid.min_poll_size,
        direction_type=hybrid.direction_type,
        display_degree=hybrid.display_degree,
        bb_max_block_size=hybrid.bb_max_block_size,
        seed=hybrid.seed + hybrid.cycle,
        x0=space.vector_from_chromosome(pso.xmin),
        max_iterations=int(hybrid.mads_iterations_per_episode),
    )
    mads.cache = hybrid.mads_cache
    mads.fxmin = float(pso.fxmin)
    mads.xmin = np.asarray(pso.xmin, dtype=int).copy()
    mads.best_vector = np.asarray(mads.x0, dtype=float)
    start_design = space.design_from_vector(mads.best_vector)
    mads.cache.setdefault(design_cache_key(start_design), float(pso.fxmin))

    phase_improved = False
    previous_best = float(mads.fxmin)
    while remaining_budget(hybrid) > 0 and int(mads.simulated_count) < int(mads.max_simulations):
        before = float(mads.fxmin)
        before_simulated = int(mads.simulated_count)
        before_total_simulated = int(hybrid.objective.simulated_count)
        before_cache_hits = int(mads.cache_hits)
        before_eval_count = int(mads.eval_count)
        before_history_count = len(mads.history_designs)
        allowed = min(int(hybrid.local_mads_budget), remaining_budget(hybrid))
        mads.max_simulations = min(int(mads.max_simulations), int(mads.simulated_count) + allowed)
        mads.initial_poll_size = hybrid.mads_frame_size
        mads.min_poll_size = hybrid.min_poll_size
        mads.max_iterations = int(hybrid.mads_iterations_per_episode)
        frame_before_episode = float(hybrid.mads_frame_size)
        print(
            f"MADS episode: frame={hybrid.mads_frame_size}, "
            f"remaining_total={remaining_budget(hybrid)}, local_used={mads.simulated_count}",
            flush=True,
        )
        callback = make_block_callback(mads, save_history=save_history)
        result = py_nomad.optimize(
            callback,
            mads.x0,
            mads.variable_space.lower_bounds,
            mads.variable_space.upper_bounds,
            nomad_parameters(mads),
        )
        mads.nomad_result = dict(result) if isinstance(result, dict) else {"raw_result": result}
        improved = mads.fxmin < before - float(hybrid.improvement_tolerance)
        made_new_simulations = int(mads.simulated_count) > before_simulated
        episode_obj = mads.history_obj[before_history_count:]
        unique_designs = count_unique_designs(mads.history_designs[before_history_count:])
        frame_after_episode = frame_before_episode
        if not improved:
            frame_after_episode = max(
                float(hybrid.min_poll_size),
                frame_before_episode * float(hybrid.mads_frame_reduction),
                1.0,
            )
        record_diagnostic_row(
            hybrid,
            phase="mads_improved" if improved else "mads_failed",
            source="mads",
            best_before=before,
            best_after=float(mads.fxmin),
            simulated_before=before_total_simulated,
            simulated_after=int(hybrid.objective.simulated_count),
            cache_hits_before=before_cache_hits,
            cache_hits_after=int(mads.cache_hits),
            callback_evaluations=int(mads.eval_count) - before_eval_count,
            unique_designs=unique_designs,
            pso_mean_npv=np.nan,
            pso_median_npv=np.nan,
            pso_current_best_npv=np.nan,
            failed_evaluations=count_failed_objectives(episode_obj),
            mads_frame_before=frame_before_episode,
            mads_frame_after=frame_after_episode,
            improved=improved,
        )
        if improved:
            phase_improved = True
            mads.x0 = mads.best_vector.tolist()
            hybrid.mads_frame_size = max(float(hybrid.mads_frame_size), float(hybrid.min_poll_size), 1.0)
            if not made_new_simulations:
                break
            continue
        hybrid.mads_frame_size = frame_after_episode
        print(
            "MADS returned to PSO after a non-improving episode; "
            f"next frame={hybrid.mads_frame_size}.",
            flush=True,
        )
        break

    hybrid.mads_cache = mads.cache
    if mads.fxmin < previous_best - float(hybrid.improvement_tolerance):
        hybrid.best_objective = float(mads.fxmin)
        hybrid.best_chromosome = np.asarray(mads.xmin, dtype=int).copy()
        hybrid.xmin = hybrid.best_chromosome.copy()
        hybrid.best_particle = particle_from_chromosome(hybrid.config, hybrid.best_chromosome)
        pso.fxmin = float(mads.fxmin)
        pso.xmin = hybrid.best_chromosome.copy()
        pso.best_particle = hybrid.best_particle.copy()
    return phase_improved


def inject_mads_best_into_pso(hybrid: HybridMADSData) -> None:
    """Inject the MADS-improved incumbent into the PSO social state."""

    pso = require_pso(hybrid)
    if hybrid.best_particle is None or hybrid.best_chromosome is None:
        return
    worst = int(np.argmax(pso.personal_best_obj))
    pso.particles[worst] = hybrid.best_particle.copy()
    pso.velocities[worst] = 0.0
    pso.personal_best_particles[worst] = hybrid.best_particle.copy()
    pso.personal_best_obj[worst] = float(hybrid.best_objective)
    pso.best_particle = hybrid.best_particle.copy()
    pso.xmin = hybrid.best_chromosome.copy()
    pso.fxmin = float(hybrid.best_objective)


def particle_from_chromosome(cfg: CaseConfig, chrom: np.ndarray) -> np.ndarray:
    """Map a decoded chromosome back to a normalized PSO particle."""

    chrom = np.asarray(chrom, dtype=int).reshape(-1)
    values: list[float] = []
    if cfg.design_var in (1, 3):
        order = chrom[: cfg.num_wells].astype(float)
        values.extend((order - 0.5) / max(float(cfg.num_wells), 1.0))
    if cfg.design_var in (1, 2):
        type_start = cfg.beforetype * cfg.num_wells
        values.extend(float(v) for v in chrom[type_start : type_start + cfg.num_wells])
        locations = decode_locations(chrom, cfg.beforeloc, cfg.num_wells, cfg.bits_per_location)
        values.extend((locations.astype(float) - 0.5) / max(float(cfg.num_locations), 1.0))
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def sync_from_pso(hybrid: HybridMADSData) -> None:
    """Copy PSO incumbent fields to the hybrid-level incumbent."""

    pso = require_pso(hybrid)
    hybrid.best_objective = float(pso.fxmin)
    hybrid.best_chromosome = None if pso.xmin is None else np.asarray(pso.xmin, dtype=int).copy()
    hybrid.xmin = None if hybrid.best_chromosome is None else hybrid.best_chromosome.copy()
    hybrid.best_particle = None if pso.best_particle is None else np.asarray(pso.best_particle, dtype=float).copy()


def record_hybrid_state(hybrid: HybridMADSData, phase: str, save_history: bool) -> None:
    """Record one hybrid progress point."""

    if hybrid.best_chromosome is None:
        return
    hybrid.phase_history.append(str(phase))
    hybrid.eval_history.append(int(hybrid.objective.simulated_count))
    hybrid.best_history.append(float(hybrid.best_objective))
    hybrid.best_chrom_history.append(hybrid.best_chromosome.copy())
    hybrid.mads_frame_history.append(float(hybrid.mads_frame_size))
    if save_history:
        save_hybrid_data(hybrid)


def record_diagnostic_row(
    hybrid: HybridMADSData,
    *,
    phase: str,
    source: str,
    best_before: float,
    best_after: float,
    simulated_before: int,
    simulated_after: int,
    cache_hits_before: int,
    cache_hits_after: int,
    callback_evaluations: float | int,
    unique_designs: float | int,
    pso_mean_npv: float,
    pso_median_npv: float,
    pso_current_best_npv: float,
    failed_evaluations: int,
    mads_frame_before: float,
    mads_frame_after: float,
    improved: bool,
) -> None:
    """Append one PSO/MADS diagnostic row for post-run analysis."""

    hybrid.diagnostic_rows.append(
        {
            "cycle": int(hybrid.cycle),
            "pso_iteration": int(hybrid.pso_iteration),
            "phase": str(phase),
            "source": str(source),
            "simulated_before": int(simulated_before),
            "simulated_after": int(simulated_after),
            "new_simulations": int(simulated_after) - int(simulated_before),
            "cache_hits_before": int(cache_hits_before),
            "cache_hits_after": int(cache_hits_after),
            "new_cache_hits": int(cache_hits_after) - int(cache_hits_before),
            "callback_evaluations": callback_evaluations,
            "unique_decoded_designs": unique_designs,
            "best_before_obj": float(best_before),
            "best_after_obj": float(best_after),
            "best_delta_obj": float(best_after) - float(best_before),
            "best_before_npv_scaled": -float(best_before),
            "best_after_npv_scaled": -float(best_after),
            "best_delta_npv_scaled": -float(best_after) + float(best_before),
            "pso_mean_npv_scaled": float(pso_mean_npv),
            "pso_median_npv_scaled": float(pso_median_npv),
            "pso_current_best_npv_scaled": float(pso_current_best_npv),
            "failed_evaluations": int(failed_evaluations),
            "mads_frame_before": float(mads_frame_before),
            "mads_frame_after": float(mads_frame_after),
            "improved": bool(improved),
        }
    )


def mean_npv_from_objectives(objv: np.ndarray | None) -> float:
    if objv is None:
        return float("nan")
    return float(np.nanmean(-np.asarray(objv, dtype=float)))


def median_npv_from_objectives(objv: np.ndarray | None) -> float:
    if objv is None:
        return float("nan")
    return float(np.nanmedian(-np.asarray(objv, dtype=float)))


def current_best_npv_from_objectives(objv: np.ndarray | None) -> float:
    if objv is None:
        return float("nan")
    return float(np.nanmax(-np.asarray(objv, dtype=float)))


def count_failed_objectives(objv) -> int:
    """Count known failed objective values in one PSO generation or MADS episode."""

    if objv is None:
        return 0
    values = np.asarray(objv, dtype=float).reshape(-1)
    return int(np.sum(np.isclose(values, 1000.0)))


def count_unique_designs(designs: list[np.ndarray]) -> int:
    """Count unique decoded designs recorded during one MADS episode."""

    if not designs:
        return 0
    return len({tuple(np.asarray(design, dtype=int).reshape(-1).tolist()) for design in designs})


def save_hybrid_data(hybrid: HybridMADSData) -> None:
    """Save hybrid PSO-MADS history to the case work directory."""

    out = optimizer_history_dir(hybrid.config)
    pso = require_pso(hybrid)
    history_chrom = (
        np.array(hybrid.best_chrom_history, dtype=int)
        if hybrid.best_chrom_history
        else np.empty((0, hybrid.config.chromosome_length), dtype=int)
    )
    history_obj = np.array(hybrid.best_history, dtype=float) if hybrid.best_history else np.empty((0,), dtype=float)
    data: dict[str, object] = {
        "method": np.array("hybrid_mads"),
        "HYBRIDphase": np.array(hybrid.phase_history),
        "HYBRIDsimulatedCount": np.array(hybrid.eval_history, dtype=int),
        "HYBRIDmadsFrame": np.array(hybrid.mads_frame_history, dtype=float),
        "HYBRIDgenb": history_chrom,
        "HYBRIDobjb": -history_obj,
        "HYBRIDtotalSimulated": np.array(int(hybrid.objective.simulated_count), dtype=int),
        "HYBRIDtotalCacheHits": np.array(int(hybrid.objective.cache_hits), dtype=int),
        "HYBRIDpbest": np.array([] if pso.personal_best_particles is None else pso.personal_best_particles, dtype=float),
        "HYBRIDpbestobj": np.array([] if pso.personal_best_obj is None else pso.personal_best_obj, dtype=float),
    }
    data.update(
        common_metadata(
            "HYBRID_MADS",
            hybrid.config,
            hybrid.cycle,
            best_chromosome=hybrid.best_chromosome,
            best_objective=hybrid.best_objective,
        )
    )
    if pso.personal_best_particles is not None and pso.personal_best_obj is not None:
        pbest_chrom = decode_lhs_population(hybrid.config, pso.personal_best_particles)
        data["HYBRIDpbestChrom"] = pbest_chrom
        data.update(best_population_payload(pbest_chrom, pso.personal_best_obj, prefix="HYBRID"))
    target = out / "tempdata.npz"
    atomic_savez(target, data, fallback_stem=f"tempdata_hybrid_cycle_{hybrid.cycle:04d}", compressed=False)
    save_hybrid_diagnostics_csv(hybrid, out / "hybrid_diagnostics.csv")


def save_hybrid_diagnostics_csv(hybrid: HybridMADSData, target) -> None:
    """Write a CSV summary of PSO and MADS decisions for post-run diagnosis."""

    if not hybrid.diagnostic_rows:
        return
    fieldnames = list(hybrid.diagnostic_rows[0].keys())
    try:
        write_diagnostics_csv(target, fieldnames, hybrid.diagnostic_rows)
    except PermissionError:
        fallback = target.with_name(f"hybrid_diagnostics_cycle_{hybrid.cycle:04d}.csv")
        write_diagnostics_csv(fallback, fieldnames, hybrid.diagnostic_rows)
        print(
            f"Could not update {target.name} because it is locked; "
            f"wrote {fallback.name} instead.",
            flush=True,
        )


def write_diagnostics_csv(target, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write diagnostics rows to one CSV file."""

    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remaining_budget(hybrid: HybridMADSData) -> int:
    """Return remaining new simulator evaluations."""

    return max(0, int(hybrid.max_simulations) - int(hybrid.objective.simulated_count))


def require_pso(hybrid: HybridMADSData) -> PSOData:
    if hybrid.pso is None:
        raise RuntimeError("PSO state has not been initialized.")
    return hybrid.pso


def print_results(hybrid: HybridMADSData) -> None:
    """Print final hybrid result."""

    print("Hybrid global-search/MADS optimization completed", flush=True)
    print(f"   Objective function for xmin: {hybrid.best_objective}", flush=True)
    print(f"   xmin: {hybrid.best_chromosome.tolist() if hybrid.best_chromosome is not None else None}", flush=True)
    print(f"   simulator evaluations: {hybrid.objective.simulated_count}", flush=True)
    print(f"   cache hits: {hybrid.objective.cache_hits}", flush=True)
