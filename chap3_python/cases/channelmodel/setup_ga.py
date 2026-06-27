"""Chapter 3 channelmodel setup.

Edit these scalars before launching a Python run.
"""


# ---------------------------------------------------------------------------
# Case paths
# ---------------------------------------------------------------------------
CASE_NAME = "channelmodel"
SOURCE_DIR = r"C:\Users\oqr7631\Documents\MATLAB\chap3\channelmodel"
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
MAXGEN = 80
POPULATION_SIZE = 50
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.01
ORDER_MUTATION_PROBABILITY = 0.03
EPSR = 1.0e-8
NUM_PARALLEL = 3


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
