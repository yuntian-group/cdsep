"""Hyperparameters for collaborative review generation experiments.

Uses the MARG/ARIES ICLR-paper benchmark (42 test papers, ~3.7 reviews/paper).
"""

N_PAPERS = 42  # full ARIES test set; LLM-judge eval is expensive but caches
N_TRAIN = 18   # optimizer train batch pool
N_VAL = 10     # held-out validation for best-iteration selection
N_TEST = 14    # held-out test set; final reported numbers
OPT_ITERATIONS = 6
BATCH_SIZE = 6
MAX_DEMOS_PER_AGENT = 2
MIPRO_TRIALS = 12
SEEDS = [42, 123, 456]
MODEL = "gpt-5.4-nano"
OPTIMIZER_MODEL = "gpt-5.4-mini"
N_WORKERS = 3
