"""
SmartSignal Engine - configuration.

Every tunable number lives here so the experiments in the report can be
reproduced by changing one file.
"""

DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

# ── Signal safety timings (seconds) ─────────────────────────────
MIN_GREEN = 10            # never give less green than this
MAX_GREEN = 30            # never give more green than this in one decision
YELLOW_TIME = 3           # mandatory yellow after every green
ALL_RED_TIME = 2          # mandatory all-red clearance after yellow
MAX_RED_WAIT = 90         # a waiting direction must be served within this
MAX_CONSECUTIVE = 2       # max green *extensions* in a row while others wait

# ── Traffic model ───────────────────────────────────────────────
SATURATION_FLOW = 1.0     # vehicles leaving per second on green
MAX_QUEUE = 40            # queue length shown as 100 % "density"
QUEUE_CAP = 80            # physical storage limit per approach

# Mean arrivals (vehicles / second) per approach.
# Deliberately unequal so a smart controller has something to learn.
ARRIVAL_RATES = {
    "PEAK":   {"NORTH": 0.22, "SOUTH": 0.16, "EAST": 0.26, "WEST": 0.10},
    "NORMAL": {"NORTH": 0.16, "SOUTH": 0.12, "EAST": 0.19, "WEST": 0.08},
    "NIGHT":  {"NORTH": 0.07, "SOUTH": 0.05, "EAST": 0.08, "WEST": 0.03},
}

# ── Priority engine (rule-based baseline, from the design doc) ──
PRIORITY_WEIGHTS = {
    "vehicles": 0.20,
    "queue": 0.25,
    "waiting": 0.25,
    "emergency": 0.25,
    "starvation": 0.05,
}
V_MAX = 40                # normalisers
Q_MAX = 40
W_MAX = 90
STARVATION_THRESHOLD = 60

# ── Reinforcement learning ──────────────────────────────────────
# RL_MODEL picks the agent the live app uses:
#   "dqn"     deep Q-network (NumPy MLP)            <- best results, default
#   "linear"  Q-learning with linear function approximation
#   "tabular" classic tabular Q-learning
RL_MODEL = "dqn"
GREEN_CHOICES = [10, 20, 30]          # action = (direction, one of these)
QUEUE_BUCKETS = [3, 8, 15, 25, 40]    # tabular state: 0-2 | 3-7 | 8-14 | 15-24 | 25-39 | 40+
STARVE_WARN = 45                      # red-wait that flags "starving" in state
ALPHA = 0.10                          # learning rate
GAMMA_PER_SECOND = 0.997              # discount per simulated second (horizon ~5 min)
EPSILON_START = 0.30                  # random exploration, decays linearly
EPSILON_END = 0.05
GUIDE_START = 0.50                    # follow priority rule (guided exploration), decays to 0
EPSILON_LIVE = 0.0                    # exploration while running the live demo
REWARD_SCALE = 100.0                  # delay (veh-seconds) / scale = cost
STARVATION_PENALTY = 2.0              # per second any queue exceeds MAX_RED_WAIT

MODEL_DIR = "models"                  # trained agents are saved here as JSON
Q_TABLE_PATH = "models/rl_tabular.json"
