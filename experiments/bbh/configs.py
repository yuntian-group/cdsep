"""Hyperparameters for BBH (Big-Bench Hard) experiments."""

N_TRAIN = 17
N_VAL = 8
N_TEST = 25
N_FEW_SHOT = 4
OPT_ITERATIONS = 8
BATCH_SIZE = 12
MAX_DEMOS = 4
MIPRO_TRIALS = 12
SEEDS = [42, 123, 456]
MODEL = "gpt-5.4-nano"
OPTIMIZER_MODEL = "gpt-5.4-mini"

TASKS = [
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_three_objects",
    "causal_judgement",
    "word_sorting",
]
