"""Hyperparameters for the synthetic underwriting experiment."""

# Sample budget. The medical-impairment cases are richer than the
# previous demographic toy, so a smaller N still has signal.
N_SAMPLES = 90
N_TRAIN = 45
N_VAL = 15
N_TEST = 30
OPT_ITERATIONS = 10
BATCH_SIZE = 15
MAX_DEMOS_PER_AGENT = 2
MIPRO_TRIALS = 12
SEEDS = [42, 123, 456]
MODEL = "gpt-5.4-nano"
OPTIMIZER_MODEL = "gpt-5.4-mini"

# Closed bucket set; matches both data.py and agents.py. Used by the
# MAE metric and by the optimizer feedback formatter.
RATING_BUCKETS = ["0", "50", "100", "150", "175", "200", "250", "300", "decline", "postpone"]

# Ordinal embedding for MAE: numeric ratings span 0-300 in steps of 50,
# decline maps above 300 (worst possible outcome), postpone slightly below
# decline. None / parse failure is treated as max distance.
RATING_TO_ORDINAL: dict[str, int] = {
    "0": 0, "50": 1, "100": 2, "150": 3, "175": 4,
    "200": 5, "250": 6, "300": 7,
    "postpone": 8, "decline": 9,
}
MAX_ORDINAL_DISTANCE = max(RATING_TO_ORDINAL.values()) - min(RATING_TO_ORDINAL.values())
