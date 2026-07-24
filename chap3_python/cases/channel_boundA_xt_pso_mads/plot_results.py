"""Plot PSO-MADS convergence and MADS trigger events."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# The optimizer updates this file after each generation.
CASE_DIR = Path(__file__).resolve().parent
TEMPDATA_FILE = CASE_DIR / "work" / "python_tempdata" / "tempdata.npz"
OUTPUT_FILE = CASE_DIR / "work" / "python_tempdata" / "pso_mads_convergence.png"

# Optional plot window for scaled NPV values.
Y_LIMITS = None

# Failed simulations are saved as objective = 1000, which becomes NPV = -1000
# after the sign flip. Hide those failures from the visible plot range.
HIDE_FAILED_VALUES = True
FAILED_OBJECTIVE_VALUE = 1000.0


def objective_keys(data: np.lib.npyio.NpzFile) -> tuple[str, str | None]:
    """Return optimizer-specific objective keys from a tempdata archive."""

    if "HYBRIDobjb" in data:
        return "HYBRIDobjb", "HYBRIDobjb"
    if "PSOobj" in data:
        return "PSOobj", "PSOobjb" if "PSOobjb" in data else None
    if "GAobj" in data:
        return "GAobj", "GAobjb" if "GAobjb" in data else None
    available = ", ".join(data.files)
    raise KeyError(f"No supported objective key found. Available keys: {available}")


def main() -> None:
    if not TEMPDATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {TEMPDATA_FILE}. Run the optimizer first, or check the case folder."
        )

    with np.load(TEMPDATA_FILE) as data:
        obj_key, best_key = objective_keys(data)
        obj = np.asarray(data[obj_key], dtype=float)
        best_so_far_npv = np.asarray(data[best_key], dtype=float) if best_key else None
        phases = np.asarray(data["HYBRIDphase"]).astype(str).reshape(-1) if "HYBRIDphase" in data else None
        simulated_count = (
            np.asarray(data["HYBRIDsimulatedCount"], dtype=int).reshape(-1)
            if "HYBRIDsimulatedCount" in data
            else None
        )

    is_hybrid = obj_key == "HYBRIDobjb"
    pop_npv = obj.copy() if is_hybrid else -obj
    if HIDE_FAILED_VALUES:
        pop_npv = np.where(np.isclose(obj, FAILED_OBJECTIVE_VALUE), np.nan, pop_npv)

    if best_so_far_npv is None:
        if pop_npv.ndim == 1:
            pop_npv = pop_npv.reshape(1, -1)
        best_new_npv = np.nanmax(pop_npv, axis=1)
        best_so_far_npv = np.maximum.accumulate(best_new_npv)
    else:
        best_so_far_npv = best_so_far_npv.reshape(-1)

    if is_hybrid:
        plot_hybrid(best_so_far_npv, phases, simulated_count)
    else:
        plot_population(pop_npv, best_so_far_npv)

    if best_so_far_npv.size:
        print(f"Final best-so-far NPV: {best_so_far_npv[-1]:.6g}")
    if phases is not None:
        print(f"MADS improved count: {np.sum(phases == 'mads_improved')}")
        print(f"MADS no-improvement count: {np.sum(phases == 'mads_failed')}")
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")
    plt.show()


def plot_hybrid(
    best_so_far_npv: np.ndarray,
    phases: np.ndarray | None,
    simulated_count: np.ndarray | None,
) -> None:
    """Plot PSO-MADS best-so-far against actual simulation count."""

    if simulated_count is None or simulated_count.size == 0:
        x_values = np.arange(best_so_far_npv.size)
        x_label = "Recorded phase"
    else:
        x_values = simulated_count[: best_so_far_npv.size]
        x_label = "Simulator evaluations"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.step(
        x_values,
        best_so_far_npv,
        where="post",
        linewidth=2.2,
        color="#8C564B",
        label="PSO-MADS best-so-far",
    )
    ax.plot(x_values, best_so_far_npv, "o", color="#8C564B", markersize=4)
    plot_mads_trigger_markers(ax, x_values, best_so_far_npv, phases)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best-so-far NPV x 10^9 USD")
    ax.set_title("PSO-MADS Convergence and MADS Triggers")
    if Y_LIMITS is not None:
        ax.set_ylim(Y_LIMITS)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()


def plot_population(pop_npv: np.ndarray, best_so_far_npv: np.ndarray) -> None:
    """Fallback population plot for non-hybrid checkpoints."""

    if pop_npv.ndim == 1:
        pop_npv = pop_npv.reshape(1, -1)
    generations = np.arange(pop_npv.shape[0])
    x = np.repeat(generations, pop_npv.shape[1])
    y = pop_npv.reshape(-1)

    fig, (ax_pop, ax_best) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_pop.plot(x, y, "o", markersize=4, alpha=0.7)
    ax_pop.set_ylabel("Population NPV x 10^9 USD")
    if Y_LIMITS is not None:
        ax_pop.set_ylim(Y_LIMITS)
    ax_pop.grid(True, alpha=0.3)

    ax_best.plot(generations, best_so_far_npv, "-o", linewidth=2, markersize=4)
    ax_best.set_xlabel("Iteration")
    ax_best.set_ylabel("Best-so-far NPV x 10^9 USD")
    if Y_LIMITS is not None:
        ax_best.set_ylim(Y_LIMITS)
    ax_best.grid(True, alpha=0.3)
    fig.tight_layout()

    finite_y = y[np.isfinite(y)]
    print(f"Plotted {finite_y.size} valid point(s).")
    if finite_y.size:
        print(f"NPV range: {np.nanmin(finite_y):.6g} to {np.nanmax(finite_y):.6g}")
        if Y_LIMITS is not None:
            hidden = np.sum((finite_y < Y_LIMITS[0]) | (finite_y > Y_LIMITS[1]))
            print(f"Point(s) outside y limits {Y_LIMITS}: {hidden}")


def plot_mads_trigger_markers(ax, x_values: np.ndarray, best_so_far_npv: np.ndarray, phases: np.ndarray | None) -> None:
    """Mark PSO-MADS local-search handoffs on one axis."""

    if phases is None:
        return
    count = min(phases.size, best_so_far_npv.size, x_values.size)
    if count == 0:
        return
    phase = phases[:count]
    x = x_values[:count]
    y = best_so_far_npv[:count]
    improved = phase == "mads_improved"
    failed = phase == "mads_failed"
    if np.any(improved):
        ax.scatter(
            x[improved],
            y[improved],
            marker="^",
            s=70,
            color="#111111",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="MADS improved",
        )
        for improved_x in x[improved]:
            ax.axvline(improved_x, color="#111111", alpha=0.12, linewidth=1.0)
    if np.any(failed):
        ax.scatter(
            x[failed],
            y[failed],
            marker="x",
            s=70,
            color="#555555",
            linewidth=1.4,
            zorder=5,
            label="MADS no improvement",
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9)


if __name__ == "__main__":
    main()
