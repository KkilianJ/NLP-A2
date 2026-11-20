"""Reuse the previous phases without altering their files or module paths."""
import importlib.util
import os
import sys

import workflow_config as cfg

os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE / "matplotlib"))
previous_path = sys.path[:]
previous_bytecode = sys.dont_write_bytecode
sys.path.insert(0, str(cfg.ASSIGNMENT / "phase3"))
sys.dont_write_bytecode = True
try:
    import legacy
    import sequence_dataset
    import sequence_model
    import sequence_preparation
    import lstm_training
    import baselines
finally:
    sys.path[:] = previous_path
    sys.dont_write_bytecode = previous_bytecode


def load_phase2_module(name):
    path = cfg.ASSIGNMENT / "phase2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"remaining_phase2_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_phase3_plotting():
    spec = importlib.util.spec_from_file_location("remaining_phase3_plots", cfg.ASSIGNMENT / "phase3/plots.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
