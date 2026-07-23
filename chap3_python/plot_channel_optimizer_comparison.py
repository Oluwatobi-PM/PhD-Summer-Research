"""Plot channel Bound A optimizer convergence comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


PROJECT_DIR = Path(__file__).resolve().parent
CASES_DIR = PROJECT_DIR / "cases"
OUTPUT_DIR = PROJECT_DIR / "figures"

METHODS = (
    ("ILHS", "#2F73D9"),
    ("PSO", "#FF7F0E"),
    ("GA", "#1CAD45"),
    ("DE", "#D62728"),
    ("MADS", "#9467BD"),
    ("PSO-MADS", "#8C564B"),
)

CASE_GROUPS = {
    "xt": {
        "ILHS": "channel_boundA_xt_ilhs",
        "PSO": "channel_boundA_xt_pso",
        "GA": "channel_boundA_xt_ga",
        "DE": "channel_boundA_xt_de",
        "MADS": "channel_boundA_xt_mads",
        "PSO-MADS": "channel_boundA_xt_pso_mads",
    },
    "xto": {
        "ILHS": "channel_boundA_xto_ilhs",
        "PSO": "channel_boundA_xto_pso",
        "GA": "channel_boundA_xto_ga",
        "DE": "channel_boundA_xto_de",
        "MADS": "channel_boundA_xto_mads",
        "PSO-MADS": "channel_boundA_xto_pso_mads",
    },
}

# Optional axis limits. Use None for automatic scaling.
# Examples:
#   AXIS_LIMITS["xt"]["y"] = (4.0e9, 5.1e9)
#   AXIS_LIMITS["xto"]["x"] = (0, 80)
AXIS_LIMITS = {
    "xt": {"x": None, "y": None},
    "xto": {"x": None, "y": None},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, cases in CASE_GROUPS.items():
        output = OUTPUT_DIR / f"channel_boundA_{label}_optimizer_comparison.png"
        plot_group(label, cases, output)
        print(f"Saved {output}")


def plot_group(label: str, cases: dict[str, str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)

    for method, color in METHODS:
        case_name = cases[method]
        tempdata = CASES_DIR / case_name / "work" / "python_tempdata" / "tempdata.npz"
        if not tempdata.exists():
            print(f"Skipping {method} for {label}: missing {tempdata}")
            continue
        best_npv = load_best_npv_usd(tempdata)
        iterations = np.arange(best_npv.size)
        ax.step(iterations, best_npv, where="post", label=method, color=color, linewidth=2.2)

    ax.set_xlabel("Iterations", fontsize=12, fontweight="bold")
    ax.set_ylabel("NPV (USD)", fontsize=12, fontweight="bold")
    limits = AXIS_LIMITS.get(label, {})
    if limits.get("x") is not None:
        ax.set_xlim(limits["x"])
    if limits.get("y") is not None:
        ax.set_ylim(limits["y"])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2E"))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9D9D9")
    ax.spines["bottom"].set_color("#D9D9D9")
    ax.tick_params(axis="both", colors="#5C5C5C", labelsize=10, length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.08), ncol=3, frameon=False, fontsize=9)
    ax.set_title(f"Channel Bound A {label.upper()} Optimization", fontsize=12, pad=16)

    fig.tight_layout(pad=2.0)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def load_best_npv_usd(tempdata: Path) -> np.ndarray:
    if not tempdata.exists():
        raise FileNotFoundError(f"Missing optimizer checkpoint: {tempdata}")

    with np.load(tempdata) as data:
        if "PSOobjb" in data:
            best_scaled = np.asarray(data["PSOobjb"], dtype=float).reshape(-1)
        elif "GAobjb" in data:
            best_scaled = np.asarray(data["GAobjb"], dtype=float).reshape(-1)
        elif "DEobjb" in data:
            best_scaled = np.asarray(data["DEobjb"], dtype=float).reshape(-1)
        elif "MADSobjb" in data:
            best_scaled = np.asarray(data["MADSobjb"], dtype=float).reshape(-1)
        elif "HYBRIDobjb" in data:
            best_scaled = np.asarray(data["HYBRIDobjb"], dtype=float).reshape(-1)
        elif "PSOobj" in data:
            best_scaled = np.maximum.accumulate(np.nanmax(-np.asarray(data["PSOobj"], dtype=float), axis=1))
        elif "GAobj" in data:
            best_scaled = np.maximum.accumulate(np.nanmax(-np.asarray(data["GAobj"], dtype=float), axis=1))
        elif "DEobj" in data:
            best_scaled = np.maximum.accumulate(np.nanmax(-np.asarray(data["DEobj"], dtype=float), axis=1))
        elif "MADSobj" in data:
            best_scaled = np.maximum.accumulate(-np.asarray(data["MADSobj"], dtype=float).reshape(-1))
        else:
            available = ", ".join(data.files)
            raise KeyError(f"No supported objective history found in {tempdata}. Keys: {available}")

    return best_scaled * 1.0e9


if __name__ == "__main__":
    main()
