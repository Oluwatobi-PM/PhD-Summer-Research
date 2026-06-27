"""Plot saved GA convergence data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_results(tempdata: str | Path, output: str | Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    tempdata = Path(tempdata)
    output = Path(output) if output else tempdata.with_name("ga_convergence.png")
    data = np.load(tempdata, allow_pickle=True)
    best_npv = data["GAobjb"]
    best_npv = best_npv[best_npv != 0]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(best_npv)), best_npv, "-o", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best NPV / 1e9")
    ax.set_title("GA Convergence")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
