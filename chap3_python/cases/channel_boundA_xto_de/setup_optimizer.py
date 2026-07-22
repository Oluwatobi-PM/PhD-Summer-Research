"""Chapter 3 channelmodel XTO setup for Differential Evolution."""

from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Case and optimizer
# ---------------------------------------------------------------------------
CASE_NAME = "channelmodel"
OPTIMIZER = "de"  # "ga", "ilhs", "pso", or "de"


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
# DE parameters
# ---------------------------------------------------------------------------
DE_OPTIONS = {
    "SEED": 3001,
    "MAX_GENERATIONS": 80,
    "DE_POPULATION_SIZE": 50,
    "DE_MUTATION_FACTOR": 0.5,
    "DE_CROSSOVER_FACTOR": 0.7,
    # UOF strategy 7 = DE/rand/1/bin, a conservative first DE baseline.
    "DE_STRATEGY": 7,
}


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
RUN_OPTIONS = {
    "INITIALIZATION": "lhs",
    "INITIALIZATION_SEED": 1000,
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
