"""Chapter 3 Brugge setup.

Edit the scalar values in this file the same way you would edit MATLAB
`setupGA.m`. The reusable optimizer code lives in `chap3_ga/`.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Case paths
# ---------------------------------------------------------------------------
CASE_NAME = "Brugge_CaseA"
SOURCE_DIR = "./work"
WORK_DIR = "./work"
TEMPLATE_DIR = "./template"


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
NUM_WELLS = 12
NUM_LOCATIONS = 30
PREF = 50.973
INJREF = 183.5715
SIM_TIME = 7300
TD = 30
# INJ_IDX_ZERO_START = 15

# DESIGN_VAR:
#   1 = optimize drilling order, well type, and location: O,T,x
#   2 = optimize well type and location: T,x
#   3 = optimize drilling order only: O
DESIGN_VAR = 2


# ---------------------------------------------------------------------------
# GA parameters
# ---------------------------------------------------------------------------
MAXGEN = 1
POPULATION_SIZE =1
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.01
ORDER_MUTATION_PROBABILITY = 0.03
EPSR = 1.0e-8
NUM_PARALLEL = 10
SIMULATION_THREADS = 1


# ---------------------------------------------------------------------------
# Optional engineering initial solutions
# ---------------------------------------------------------------------------
# Leave this empty for a fully random initial population.
# If POPULATION_SIZE = 5 and you add one solution here, that solution becomes
# population row 1 and the other four rows are generated randomly.
#
# For DESIGN_VAR = 2, use:
#   types:     0 = producer, 1 = injector
#   locations: 1-based candidate-location labels
INITIAL_SOLUTIONS = [
    {
        "types": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "locations": [29, 14, 16, 13, 26, 23, 5, 17, 12, 21, 11, 19],
    },
]

# If you already have the fully encoded GA chromosome, use this instead.
# The row length must match the setup's chromosome_length from --check-setup.
INITIAL_CHROMOSOMES = [
    # [0, 1, 0, 0, 0, 1],  # paste the full encoded chromosome row here
]


# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
DRY_RUN = False
CHECK_SETUP_ONLY = False
SEED = 1000
STREAM_SIMULATOR_OUTPUT = True
PRINT_BATCH_TIMING = True
RESULTS_TIMEOUT_SECONDS = 60
SIMULATION_INTERRUPT_TIMEOUT_SECONDS = 60

# After a type/location run, update the order-only input table
# `work/baseinfo1_locidx.csv`. This is kept on for both real and dry runs.
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
    from chap3_ga.run_case import run_from_setup

    run_from_setup(__file__)
