"""Read-only access to Phase 2's verified split/metrics and Phase 1 tokenizer."""
import os
import sys

import experiment_settings as cfg

os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE_DIR / "matplotlib"))
_previous_path = sys.path[:]
_previous_bytecode = sys.dont_write_bytecode
sys.path.insert(0, str(cfg.PHASE2_ROOT))
sys.dont_write_bytecode = True
try:
    from data import load_partitions, indices_sha
    from metrics import evaluate
    from settings import THRESHOLD as phase2_threshold
    from phase1_imports import phase1, phase1_config
    if phase2_threshold != cfg.THRESHOLD:
        raise ValueError("Phase 2 metric threshold differs from the required Phase 3 threshold.")
finally:
    # The legacy import also modifies sys.path. Restore it so Phase 3's plots.py
    # and other modules cannot accidentally resolve to Phase 2's namesakes.
    sys.path[:] = _previous_path
    sys.dont_write_bytecode = _previous_bytecode
