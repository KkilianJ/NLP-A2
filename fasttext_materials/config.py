"""All substantive Phase 1 choices; paths are relative to this directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "cc.en.300.bin"
MODEL_SHA256 = "14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca"
CACHE_DIR = ROOT / ".cache"
SEED = 42
DATASET_REVISION = "d05f0155defa7991dad75bc68c5ccb6774b1fdc5"
ORDERED_DATA_SHA256 = "0ed0de756c57a4ac8868e7cc2a9970650e930d94cdb0347716d385c9e1e705be"
SOURCE_SHA256 = {
    "train.parquet": "541418c6fe070c2a82b31e786581605d2db531c7cb34baa02b86b4576a21d4dc",
    "test.parquet": "6f16f9843933504caf9c17410cbeb0449b55b72bbbccbaf59d5544a4999d367b",
}
SMOKE_REVIEWS = 200
NEIGHBOR_CANDIDATES = 100_000  # first N words in the model's stored frequency order
SMOKE_NEIGHBOR_CANDIDATES = 5_000
NEIGHBORS_PER_QUERY = 5
MIN_CORRECTION_REVIEW_FREQUENCY = 20
CORRECTION_OOV_LIMIT = 1_500
CONTEXT_EXAMPLES_PER_CATEGORY = 12

# Each pair is deliberately easy to inspect, rather than a benchmark or a learned lexicon.
SEMANTIC_PAIRS = {
    "positive-positive": [("excellent", "wonderful"), ("amazing", "fantastic"),
                          ("good", "great"), ("beautiful", "lovely"),
                          ("enjoyable", "entertaining"), ("brilliant", "excellent")],
    "negative-negative": [("terrible", "awful"), ("horrible", "awful"), ("bad", "poor"),
                          ("boring", "dull"), ("disappointing", "mediocre"),
                          ("dreadful", "horrible")],
    "positive-negative": [("good", "bad"), ("excellent", "terrible"),
                          ("wonderful", "awful"), ("enjoyable", "boring"),
                          ("beautiful", "ugly"), ("love", "hate")],
    "unrelated controls": [("excellent", "potato"), ("terrible", "planet"),
                           ("wonderful", "keyboard"), ("boring", "river"),
                           ("amazing", "bicycle"), ("awful", "umbrella")],
    "morphological controls": [("enjoy", "enjoyable"), ("beautiful", "beautifully"),
                               ("bored", "boring"), ("hope", "hopeless"),
                               ("good", "goodness"), ("bad", "badly")],
}
SPELLING_WORDS = ["excellent", "amazing", "wonderful", "brilliant", "enjoyable", "beautiful",
                  "great", "fantastic", "terrible", "horrible", "awful", "boring",
                  "disappointing", "dreadful", "bad", "waste", "acting", "cinema", "movie", "story"]
SPELLING_CONTROLS = ["potato", "keyboard", "planet", "river", "bicycle", "umbrella", "chemistry", "airport"]
NEIGHBOR_WORDS = ["excellent", "wonderful", "good", "bad", "terrible", "boring"]
TYPO_NEIGHBOR_BASES = ["excellent", "terrible", "boring"]
TYPO_NEIGHBOR_TYPES = ["deletion", "repetition"]
