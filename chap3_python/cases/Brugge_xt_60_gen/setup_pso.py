"""Chapter 3 Brugge PSO setup for the x,T stage.

PSO particles move in normalized [0, 1] space and are decoded to the same
type/location design representation used by the shared objective pipeline.
"""

from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Case and optimizer
# ---------------------------------------------------------------------------
CASE_NAME = "Brugge_CaseA"
OPTIMIZER = "pso"


# ---------------------------------------------------------------------------
# Case paths
# ---------------------------------------------------------------------------
SOURCE_DIR = "./source"
WORK_DIR = "./work"
TEMPLATE_DIR = SOURCE_DIR


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
NUM_WELLS = 12
NUM_LOCATIONS = 30
PREF = 50.973
INJREF = 183.5715
SIM_TIME = 7300
TD = 30

# DESIGN_VAR:
#   1 = optimize drilling order, well type, and location: O,T,x
#   2 = optimize well type and location: T,x
#   3 = optimize drilling order only: O
DESIGN_VAR = 2


# ---------------------------------------------------------------------------
# PSO parameters
# ---------------------------------------------------------------------------
PSO_OPTIONS = {
    "SEED": 1000,
    "MAX_ITERATIONS": 40,
    "SWARM_SIZE": 50,
    "OMEGA": 0.7298,
    "PHIP": 1.496,
    "PHIG": 1.496,
    # The paper description starts PSO at rest; use "random" to mimic the
    # starter pyswarm scripts' initial random velocity.
    "INITIAL_VELOCITY": "zero",
}


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
RUN_OPTIONS = {
    "INITIALIZATION": "lhs",
    "INITIALIZATION_SEED": 1000,
    "NUM_PARALLEL": 10,
    "SIMULATION_THREADS": 1,
    "DRY_RUN": True,
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
OIL_PRICE = 80.0
WATER_PRODUCTION_COST = 5.0
WATER_INJECTION_COST = 5.0
DISCOUNT_FACTOR = 0.1
CDRILL_V = 8.0e6
OBJECTIVE_SCALING = 1.0e9


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from chap3_pso.run_case import run_from_setup

    run_from_setup(__file__)
