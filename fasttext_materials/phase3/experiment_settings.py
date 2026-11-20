"""Default values and paths for the single, unidirectional LSTM baseline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = ROOT.parent
PHASE2_ROOT = ASSIGNMENT_ROOT / "phase2"
DATA_DIR = ASSIGNMENT_ROOT / "data"
MODEL_PATH = ASSIGNMENT_ROOT / "models" / "cc.en.300.bin"
BASELINE_CSV = PHASE2_ROOT / "results" / "seed-42_epochs-30" / "comparison.csv"
LINEAR_BASELINE_DIR = ROOT / "results" / "lstm-s42-h64-l512-b32-lr0.001-e30"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"
DIMENSION = 300
HIDDEN_SIZE = 64
HEAD_HIDDEN_SIZE = 64
MLP_MAX_LENGTH = 512
MAX_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
SEED = 42
SPLIT_SEED = 42
EARLY_STOP_PATIENCE = 5
THRESHOLD = 0.5
SMOKE_TRAIN_REVIEWS = 64
SMOKE_VALIDATION_REVIEWS = 32
CACHE_VERSION = 1
PROGRESS_EVERY = 500
