# Chapter 3 Python GA Port

This folder is a Python port of the MATLAB workflow in:

`C:\Users\oqr7631\Documents\MATLAB\chap3`

The MATLAB control code has been converted into a Python package. The CMG simulator inputs, case folders, `.mat` files, and `.bat` launchers remain external model assets and are referenced from the original folder by default.

## What Was Converted

- `setupGA.m` -> `chap3_ga.config`
- `GA_base_struct.m`, `GA_opt.m`, `gaiteration.m`, `garesults.m` -> `chap3_ga.ga`
- `evalObjective.m` -> `chap3_ga.objective`
- `writeloc*.m`, `writesch*.m` -> `chap3_ga.writers`
- `plot_results.m` -> `chap3_ga.plotting`
- `save_gen_data.m` -> JSON/NPZ generation snapshots in the run folder

## Preferred Structure

The preferred workflow is now case-folder based. Each optimization problem
should have one setup file, one reusable simulator template, and one working
directory for numbered parallel runs:

```text
chap3_python/
  cases/
    Brugge_CaseA/
      setup.py
      template/
      work/
    channelmodel/
      setup.py
      template/
      work/
  chap3_ga/
    reusable optimizer/objective/writer code
  chap3_ilhs/
    reusable ILHS optimizer code
```

Edit the scalar values in a case's setup file, similar to MATLAB `setupGA.m`.
Set `OPTIMIZER = "ga"` or `OPTIMIZER = "ilhs"` in that file. Shared case and
model settings stay as top-level scalars; optimizer/testing switches can live
in `GA_OPTIONS`, `ILHS_OPTIONS`, `RUN_OPTIONS`, and `OBJECTIVE_OPTIONS`
dictionaries. The selected dictionary is overlaid onto the setup at load time,
so older flat variables such as `MAXGEN`, `DRY_RUN`, and `NUM_PARALLEL` still
work.

`template/` is copied to `work/1`, `work/2`, ... up to `NUM_PARALLEL`. The
generic code in `chap3_ga/` and `chap3_ilhs/` should not need case-specific
parameter edits.

## Run

From this folder:

```powershell
python -m chap3_ga.opt_cli --setup ".\cases\channelmodel\setup.py" --check-setup
python -m chap3_ga.opt_cli --setup ".\cases\channelmodel\setup.py"
```

## Seed Initial Chromosomes

To include engineering-heuristic solutions in the initial GA population, add
one of these optional blocks to a case `setup_ga.py`.

Raw encoded chromosomes:

```python
INITIAL_CHROMOSOMES = [
    [0, 1, 0, 0, 0, 1],  # paste the full chromosome row
]
```

Readable solutions for `DESIGN_VAR = 2` (`T,x`):

```python
INITIAL_SOLUTIONS = [
    {
        "types": [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        "locations": [29, 17, 20, 26, 13, 11, 16, 7, 18, 1, 12, 28],
    },
]
```

Readable solutions for `DESIGN_VAR = 1` (`O,T,x`) also include `"order"`.
Readable solutions for `DESIGN_VAR = 3` (`O`) only need `"order"`.

If `POPULATION_SIZE = 5` and one solution is provided, the first population
member is your solution and the other four are generated randomly. If
`POPULATION_SIZE = 1` and `MAXGEN = 1`, the run evaluates only the provided
solution.

The CLI also accepts a JSON/CSV/TXT file:

```powershell
python -m chap3_ga.cli --setup ".\cases\Brugge_xt_o\setup_ga.py" --initial-chromosomes ".\seed.csv"
```

## Restart From A Saved Population

To continue an existing run from the last saved population, use its
`work/python_tempdata/tempdata.npz` file as a restart checkpoint. Prefer a new
work directory when testing so the original run history stays intact.

```powershell
python -m chap3_ga.cli `
  --setup ".\cases\Brugge_xt_o\setup_ga.py" `
  --restart-from ".\cases\Brugge_xt_o\work\python_tempdata\tempdata.npz" `
  --extra-generations 20 `
  --work-dir ".\cases\Brugge_xt_o_restart20\work"
```

The restart loads the last saved population, preserves the previous history in
the new checkpoint, and appends the additional generations.

## Check Setup Before Running

Use `--check-setup` to verify which `.mat` file is loaded and whether the
required simulator folders exist. This does not run the GA or CMG.

```powershell
python -m chap3_ga.cli --case Brugge_CaseA_xt_o --source "C:\Users\oqr7631\Documents\MATLAB\chap3\Brugge_CaseA_xt_o" --design-var 1 --check-setup
```

MAT-file selection follows the original `setupGA.m` logic:

- `design_var 1`: loads `baseinfo.mat`
- `design_var 2`: loads `baseinfo.mat`
- `design_var 3`: loads `baseinfo1.mat` for Brugge cases, `baseinfo.mat` for `channelmodel`

Other supported cases:

```powershell
python -m chap3_ga.cli --setup ".\cases\Brugge\setup_ga.py"
python -m chap3_ga.cli --setup ".\cases\channelmodel\setup_ga.py"
```

By default this runs the full GA. For a quick smoke test without launching CMG:

```powershell
python -m chap3_ga.cli --setup ".\cases\channelmodel\setup_ga.py" --dry-run --maxgen 1 --np 3
```

## Notes

- MATLAB arrays are 1-based; the Python implementation stores labels as normal Python integers but preserves the MATLAB chromosome encoding.
- The Python objective runner uses `subprocess.run` instead of MATLAB `system`/`parfor`.
- `saveas2.m` was a MATLAB figure-saving utility. In Python, use normal `matplotlib` `savefig`.
- The original MATLAB schedule writers used `locidex(i)` in a loop over `j`; the Python version uses the current well index, which matches the surrounding intent.
