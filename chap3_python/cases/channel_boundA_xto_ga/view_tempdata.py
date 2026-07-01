"""Inspect this case's optimizer checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parent
CHAP3_ROOT = CASE_ROOT.parents[1]
sys.path.insert(0, str(CHAP3_ROOT))

from chap3_ga.view_tempdata import main  # noqa: E402


if __name__ == "__main__":
    main(CASE_ROOT)
