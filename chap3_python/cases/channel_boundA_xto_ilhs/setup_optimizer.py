"""Chapter 3 channelmodel setup.

This file follows the general case pattern:

- `SOURCE_DIR` contains reusable case data and model libraries.
- `SOURCE_DIR` is copied to `WORK_DIR/1`, `WORK_DIR/2`, ... for parallel runs.
- `OPTIMIZER` selects which optimizer engine runs this case.
"""

from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Case and optimizer
# ---------------------------------------------------------------------------
CASE_NAME = "channelmodel"
OPTIMIZER = "ilhs"  # "ga" or "ilhs"


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
# GA parameters
# ---------------------------------------------------------------------------
GA_OPTIONS = {
    "SEED": 2001,
    "MAXGEN": 80,
    "POPULATION_SIZE": 50,
    "CROSSOVER_PROBABILITY": 0.9,
    "MUTATION_PROBABILITY": 0.01,
    "ORDER_MUTATION_PROBABILITY": 0.03,
    "EPSR": 1.0e-8,
}


# ---------------------------------------------------------------------------
# ILHS parameters
# ---------------------------------------------------------------------------
ILHS_OPTIONS = {
    "SEED": 3001,
    "MAX_ITERATIONS": 80,
    "NUMBER_OF_SAMPLES": 50,
    "ENTROPY": 0.9,
}


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
RUN_OPTIONS = {
    "INITIALIZATION": "lhs",
    "INITIALIZATION_SEED": 1000,
    "NUM_PARALLEL": 7,
    "SIMULATION_THREADS": 1,
    "DRY_RUN": True,
    "CHECK_SETUP_ONLY": False,
    "CLEAN_WORK_FOLDERS_ON_START": True,
    "CLEAN_HISTORY_ON_START": True,
    "STREAM_SIMULATOR_OUTPUT": True,
    "PRINT_BATCH_TIMING": True,
    "RESULTS_TIMEOUT_SECONDS": 60,
    "SIMULATION_INTERRUPT_TIMEOUT_SECONDS": 60,
    "UPDATE_BASEINFO1_AFTER_RUN": True,
    "ALLOW_DRY_RUN_BASEINFO1_UPDATE": True,
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
