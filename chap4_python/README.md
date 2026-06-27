# Chapter 4 Python Port

This folder ports the MATLAB workflow in:

`C:\Users\oqr7631\Documents\MATLAB\chap4`

The original MATLAB and simulator folders are left untouched. The Python code can copy numbered simulator folders into a separate work directory, generate the same schedule/location files, and either run a dry objective for testing or launch the external simulator batch files.

## Supported Cases

- `Brugge`: CMG-style workflow using `RunCMG.bat`, `waterFlooding.rwo`, `waterFlooding_well_location.inc`, and `waterFlooding_sched.inc`
- `PUNQ`: Eclipse-style workflow using `RunEclipse.bat`, `PUN_E100.RSM`, and `PUN_SCH.inc`

## Supported Algorithms

- `MixencodeGA`
- `GenocopIII`
- `Iterative`

The iterative mode preserves the GA path and variable-forming behavior. The expensive StoSAG/GPS local-search routines are represented as Python extension points because they depend heavily on repeated simulator calls.

## Run A Dry Smoke Test

Dry mode does **not** run CMG or Eclipse. It verifies that the Python code can load case data, generate chromosomes, write simulator input files, and run a GA generation.

```powershell
cd "C:\Users\oqr7631\Documents\PhD summer research\chap4_python"
python -m chap4_opt.cli --case Brugge --source "C:\Users\oqr7631\Documents\MATLAB\chap4\Brugge" --dry-run
python -m chap4_opt.cli --case PUNQ --source "C:\Users\oqr7631\Documents\MATLAB\chap4\PUNQ" --dry-run
```

## Run With Simulator

Remove `--dry-run`. The external simulator executables/batch files must work on the machine.

```powershell
python -m chap4_opt.cli --case PUNQ --source "C:\Users\oqr7631\Documents\MATLAB\chap4\PUNQ"
```

## Notes

- The combined decision vector stores each well as `[active, type, i, j]`.
- Type `0` is intended producer, type `1` is intended injector. For active wells in non-oil gridblocks, writers force injector behavior as in MATLAB.
- Generation history is saved to `python_tempdata/tempdata.npz`.
