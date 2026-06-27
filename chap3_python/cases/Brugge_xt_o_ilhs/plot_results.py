"""Plot GA optimization values from the Python checkpoint file."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# The optimizer updates this file after each generation.
CASE_DIR = Path(__file__).resolve().parent
TEMPDATA_FILE = CASE_DIR / "work" / "python_tempdata" / "tempdata.npz"

# MATLAB-style plot window for real Brugge NPV values.
Y_LIMITS = (0, 10)

# Failed simulations are saved as objective = 1000, which becomes NPV = -1000
# after the sign flip. Hide those failures from the visible plot range.
HIDE_FAILED_VALUES = True
FAILED_OBJECTIVE_VALUE = 1000.0


def main() -> None:
    if not TEMPDATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {TEMPDATA_FILE}. Run the optimizer first, or check the case folder."
        )

    # GAobj has the minimized objective values. The MATLAB code plots NPV as
    # -GAobj, so we do the same here.
    with np.load(TEMPDATA_FILE) as data:
        gaobj = np.asarray(data["GAobj"], dtype=float)
    pop_npv = -gaobj
    if HIDE_FAILED_VALUES:
        pop_npv = np.where(np.isclose(gaobj, FAILED_OBJECTIVE_VALUE), np.nan, pop_npv)

    # GAobj is saved as generations x population. If only one generation exists,
    # keep it two-dimensional so the plotting code stays consistent.
    if pop_npv.ndim == 1:
        pop_npv = pop_npv.reshape(1, -1)

    generations = np.arange(pop_npv.shape[0])
    x = np.repeat(generations, pop_npv.shape[1])
    y = pop_npv.reshape(-1)

    plt.figure()
    plt.plot(x, y, "o")
    plt.xlabel("Generation")
    plt.ylabel("NPV x 10^9 USD")
    if Y_LIMITS is not None:
        plt.ylim(Y_LIMITS)
    plt.grid(True, alpha=0.3)
    finite_y = y[np.isfinite(y)]
    print(f"Plotted {finite_y.size} valid point(s).")
    if finite_y.size:
        print(f"NPV range: {np.nanmin(finite_y):.6g} to {np.nanmax(finite_y):.6g}")
        if Y_LIMITS is not None:
            hidden = np.sum((finite_y < Y_LIMITS[0]) | (finite_y > Y_LIMITS[1]))
            print(f"Point(s) outside y limits {Y_LIMITS}: {hidden}")
    plt.show()


if __name__ == "__main__":
    main()
