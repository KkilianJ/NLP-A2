"""Step 01 configuration. Phase 1 data and model identities remain authoritative."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = ROOT.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
SEED = 42  # example selection only
PROJECT_SPLIT_SEED = 42
PROGRESS_EVERY = 500
EXCERPT_RADIUS = 45
EXPECTED_TRAIN_COUNT = 34_707
EXPECTED_SOURCE_TRAIN_COUNT = 24_904
EXPECTED_TRAIN_IDS_SHA256 = "3fe569c54add383b4b7249ea3465297678599f30dc8c0d48d43ee7186f09ec82"
VARIANTS = ("whitespace", "regex", "regex_negation")

# Expand only complete, unambiguous negative auxiliary forms. Curly apostrophes
# are accepted too; unknown, ambiguous and multiply contracted forms stay intact.
NEGATIONS = {
    "don't": "do", "doesn't": "does", "didn't": "did",
    "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
    "can't": "can", "couldn't": "could", "won't": "will", "wouldn't": "would",
    "shouldn't": "should", "mustn't": "must", "needn't": "need",
    "hasn't": "has", "haven't": "have", "hadn't": "had",
    "mightn't": "might", "shan't": "shall",
}
EXAMPLE_NEGATIONS = ("don't", "isn't", "wasn't", "can't", "won't")
