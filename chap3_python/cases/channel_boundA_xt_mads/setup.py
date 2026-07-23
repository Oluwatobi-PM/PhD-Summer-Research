"""Run the channel_boundA_xt_mads optimizer setup."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


_setup_path = Path(__file__).with_name("setup_optimizer.py")
_spec = importlib.util.spec_from_file_location("channel_boundA_xt_mads_setup_optimizer", _setup_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load setup optimizer from {_setup_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chap3_ga.optimizer import run_from_setup

    run_from_setup(__file__)

