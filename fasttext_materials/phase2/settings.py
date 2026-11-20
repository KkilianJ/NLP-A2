"""Phase 2 experiment settings; the split seed remains fixed by Phase 1."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = ROOT.parent
DEFAULT_DATA_DIR = ASSIGNMENT_ROOT / "data"
DEFAULT_MODEL = ASSIGNMENT_ROOT / "models" / "cc.en.300.bin"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"
SPLIT_SEED = 42
SEED = 42
DIMENSION = 300
LR_C = 1.0
LR_MAX_ITER = 1000
MLP_HIDDEN = 64
MLP_LR = 0.001
MLP_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512  # inference only; does not change the training batch size
MLP_MAX_EPOCHS = 30
MLP_WEIGHT_DECAY = 0.0
THRESHOLD = 0.5
TOKEN_VECTOR_CACHE_SIZE = 50_000
EXTRACT_PROGRESS_EVERY = 500
CACHE_VERSION = 1
EXPERIMENTS = {
    "lr-mean": ("Logistic Regression", "mean"),
    "lr-max": ("Logistic Regression", "max"),
    "mlp-mean": ("MLP (64)", "mean"),
    "mlp-max": ("MLP (64)", "max"),
}
