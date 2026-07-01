"""Unified setup entry point for the channelmodel case."""

from pathlib import Path
import importlib.util

_setup_path = Path(__file__).with_name("setup_ga.py")
_spec = importlib.util.spec_from_file_location("channelmodel_setup_ga", _setup_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {_setup_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name, _value in vars(_module).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chap3_ga.optimizer import run_from_setup

    run_from_setup(__file__)
