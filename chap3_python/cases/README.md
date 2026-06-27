# Chapter 3 Case Setups

Each subfolder should contain a setup file, a reusable simulator `template/`,
and a generated `work/` directory. The setup file is the Python equivalent of
MATLAB `setupGA.m`: edit the scalar values at the top, set `OPTIMIZER = "ga"`
or `OPTIMIZER = "ilhs"`, then run the generic optimizer CLI with `--setup`.

Example:

```powershell
python -m chap3_ga.opt_cli --setup ".\cases\channelmodel\setup.py" --check-setup
python -m chap3_ga.opt_cli --setup ".\cases\channelmodel\setup.py"
```

The reusable code in `chap3_ga/` should not need case-specific edits.

When a case has `TEMPLATE_DIR = SOURCE_DIR`, Python will create missing
numbered run folders in `WORK_DIR` by copying the permanent input folder:

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
source/baseinfo1_locidx.csv
```

If those CSV files are absent, the code falls back to the original MATLAB
`.mat` files.
