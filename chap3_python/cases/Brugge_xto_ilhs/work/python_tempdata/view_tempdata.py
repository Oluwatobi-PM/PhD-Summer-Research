"""Inspect the optimizer checkpoint in the current python_tempdata folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np


TEMPDATA_FILE = Path(__file__).resolve().with_name("tempdata.npz")
FAILED_OBJECTIVE_VALUE = 1000.0


def format_array_preview(array: np.ndarray, max_rows: int = 8) -> str:
    """Return a compact preview for an array without flooding the terminal."""

    if array.ndim == 0:
        return str(array.item())

    if array.size == 0:
        return "[]"

    if array.ndim == 1:
        preview = array[:max_rows]
    else:
        preview = array.reshape(array.shape[0], -1)[:max_rows]

    text = np.array2string(preview, precision=6, suppress_small=False)
    if array.shape[0] > max_rows:
        text += f"\n... ({array.shape[0] - max_rows} more row(s))"
    return text


def print_key_summary(name: str, array: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")

    if np.issubdtype(array.dtype, np.number) and array.size:
        finite = array[np.isfinite(array)]
        if finite.size:
            print(f"min: {np.nanmin(finite):.10g}")
            print(f"max: {np.nanmax(finite):.10g}")
            print(f"mean: {np.nanmean(finite):.10g}")

    print("preview:")
    print(format_array_preview(array))


def print_objective_summary(data: np.lib.npyio.NpzFile) -> None:
    if "GAobj" not in data:
        return

    gaobj = np.asarray(data["GAobj"], dtype=float)
    if gaobj.ndim == 1:
        gaobj = gaobj.reshape(1, -1)

    print("\nObjective summary")
    print("-----------------")
    print("GAobj is minimized objective. NPV is shown as -GAobj.")
    print("iter  best_new_npv  mean_new_npv  worst_new_npv  failed")

    for iteration, row in enumerate(gaobj):
        failed = np.isclose(row, FAILED_OBJECTIVE_VALUE)
        valid = row[~failed]
        if valid.size == 0:
            print(f"{iteration:4d}  no valid simulations  failed={failed.sum()}")
            continue

        npv = -valid
        print(
            f"{iteration:4d}  "
            f"{np.nanmax(npv):12.6f}  "
            f"{np.nanmean(npv):12.6f}  "
            f"{np.nanmin(npv):13.6f}  "
            f"{failed.sum():6d}"
        )

    if "GAobjb" in data:
        best = np.asarray(data["GAobjb"], dtype=float).reshape(-1)
        print("\nBest-so-far NPV by iteration:")
        print(np.array2string(best, precision=6, suppress_small=False))


def print_best_chromosome(data: np.lib.npyio.NpzFile) -> None:
    if "GAobj" not in data or "GAgen" not in data:
        return

    gaobj = np.asarray(data["GAobj"], dtype=float)
    chrom = np.asarray(data["GAgen"])
    if gaobj.ndim == 1:
        gaobj = gaobj.reshape(1, -1)
        chrom = chrom.reshape(1, chrom.shape[0], -1)

    best_flat = int(np.nanargmin(gaobj))
    best_iter, best_sample = np.unravel_index(best_flat, gaobj.shape)

    print("\nBest sampled chromosome")
    print("-----------------------")
    print(f"iteration: {best_iter}")
    print(f"sample: {best_sample}")
    print(f"objective: {gaobj[best_iter, best_sample]:.10g}")
    print(f"NPV: {-gaobj[best_iter, best_sample]:.10g}")
    print(np.array2string(chrom[best_iter, best_sample], max_line_width=120))


def main() -> None:
    if not TEMPDATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {TEMPDATA_FILE}")

    print(f"Reading: {TEMPDATA_FILE}")
    with np.load(TEMPDATA_FILE, allow_pickle=False) as data:
        print("\nKeys:")
        for key in data.files:
            print(f"  - {key}")

        print_objective_summary(data)
        print_best_chromosome(data)

        print("\nArray contents")
        print("==============")
        for key in data.files:
            print_key_summary(key, np.asarray(data[key]))


if __name__ == "__main__":
    main()
