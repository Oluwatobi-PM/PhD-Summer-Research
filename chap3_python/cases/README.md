# Chapter 3 Case Setups

Each subfolder contains a `setup_ga.py` file. This is the Python equivalent of
MATLAB `setupGA.m`: edit the scalar values at the top, then run the optimizer
with `--setup`.

Example:

```powershell
python -m chap3_ga.cli --setup ".\cases\Brugge\setup_ga.py" --check-setup
python -m chap3_ga.cli --setup ".\cases\Brugge\setup_ga.py" --dry-run --maxgen 1 --np 4 --num-parallel 2
```

The reusable code in `chap3_ga/` should not need case-specific edits.

If a case has `TEMPLATE_DIR = "./template"`, Python will create missing
numbered run folders in `WORK_DIR` by copying that one template folder:

```text
template/  ->  work/1
template/  ->  work/2
template/  ->  work/3
...
```

Existing `work/N` folders are left alone.

Case data is read from CSV first. For the Chapter 3 Brugge case:

```text
work/baseinfo_locidx.csv
work/baseinfo1_locidx.csv
```

If those CSV files are absent, the code falls back to the original MATLAB
`.mat` files.
