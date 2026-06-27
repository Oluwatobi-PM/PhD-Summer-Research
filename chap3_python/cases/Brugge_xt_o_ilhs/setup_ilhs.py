"""Chapter 3 Brugge ILHS setup.

This runs Iterative Latin Hypercube Sampling for the same Brugge design
variables used by the GA workflow.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Case paths
# ---------------------------------------------------------------------------
CASE_NAME = "Brugge_CaseA"
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
# ILHS parameters
# ---------------------------------------------------------------------------
# NUMBER_OF_SAMPLES is comparable to GA population size.
# MAX_ITERATIONS is comparable to GA number of generations.
NUMBER_OF_SAMPLES = 50
MAX_ITERATIONS = 40
ENTROPY = 0.9


NUM_PARALLEL = 20
SIMULATION_THREADS = 1


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
DRY_RUN = False
CHECK_SETUP_ONLY = False
SEED = 1000
CLEAN_WORK_FOLDERS_ON_START = True
CLEAN_HISTORY_ON_START = True
STREAM_SIMULATOR_OUTPUT = True
PRINT_BATCH_TIMING = True
RESULTS_TIMEOUT_SECONDS = 60
SIMULATION_INTERRUPT_TIMEOUT_SECONDS = 60

# After a type/location run, update the order-only input table
# `source/baseinfo1_locidx.csv`. This is kept on for both real and dry runs.
UPDATE_BASEINFO1_AFTER_RUN = True
ALLOW_DRY_RUN_BASEINFO1_UPDATE = True


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
    from chap3_ilhs.run_case import run_from_setup

    run_from_setup(__file__)
