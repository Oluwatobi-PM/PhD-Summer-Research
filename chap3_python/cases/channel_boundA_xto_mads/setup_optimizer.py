"""Chapter 3 channelmodel XTO setup for standalone NOMAD/MADS."""

from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Case and optimizer
# ---------------------------------------------------------------------------
CASE_NAME = "channelmodel"
OPTIMIZER = "mads"  # "ga", "ilhs", "pso", "de", or "mads"


# ---------------------------------------------------------------------------
# Case paths
# ---------------------------------------------------------------------------
SOURCE_DIR = "./source"
TEMPLATE_DIR = SOURCE_DIR
WORK_DIR = "./work"


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
NUM_WELLS = 8
NUM_LOCATIONS = 30
PREF = 3000
INJREF = 4600
SIM_TIME = 4000
TD = 90
VERT_IDX_ZERO_START = 12

#   1 = O,T,x
#   2 = T,x
#   3 = O
DESIGN_VAR = 1


# ---------------------------------------------------------------------------
# NOMAD/MADS parameters
# ---------------------------------------------------------------------------
MADS_OPTIONS = {
    "SEED": 3001,
    "MAX_SIMULATIONS": 4000,
    "INITIAL_MESH_SIZE": 0.25,
    "INITIAL_POLL_SIZE": 0.25,
    "MIN_MESH_SIZE": 0.01,
    "MIN_POLL_SIZE": 0.01,
    "DIRECTION_TYPE": "ORTHO 2N",
    "BB_MAX_BLOCK_SIZE": 20,
    "DISPLAY_DEGREE": 0,
}


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
RUN_OPTIONS = {
    "NUM_PARALLEL": 20,
    "SIMULATION_THREADS": 1,
    "DRY_RUN": False,
    "CHECK_SETUP_ONLY": False,
    "CLEAN_WORK_FOLDERS_ON_START": True,
    "CLEAN_HISTORY_ON_START": True,
    "STREAM_SIMULATOR_OUTPUT": True,
    "PRINT_BATCH_TIMING": True,
    "RESULTS_TIMEOUT_SECONDS": 60,
    "SIMULATION_INTERRUPT_TIMEOUT_SECONDS": 60,
    "UPDATE_BASEINFO1_AFTER_RUN": False,
    "ALLOW_DRY_RUN_BASEINFO1_UPDATE": False,
}


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------
OIL_PRICE = 50.0
WATER_PRODUCTION_COST = 5.0
WATER_INJECTION_COST = 5.0
DISCOUNT_FACTOR = 0.1
CDRILL_V = 8.0e6
CDRILL_H = 1.6e7
OBJECTIVE_SCALING = 1.0e9


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chap3_ga.optimizer import run_from_setup

    run_from_setup(__file__)
