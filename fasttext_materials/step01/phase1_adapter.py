"""Read Phase 1 utilities without changing its files or import search path."""
import os
import sys

import settings

os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(settings.CACHE_DIR / "matplotlib"))
previous_path = sys.path[:]
previous_bytecode = sys.dont_write_bytecode
sys.path.insert(0, str(settings.ASSIGNMENT_ROOT))
sys.dont_write_bytecode = True
try:
    import run_phase1 as phase1
finally:
    sys.path[:] = previous_path
    sys.dont_write_bytecode = previous_bytecode

if phase1.cfg.ROOT.resolve() != settings.ASSIGNMENT_ROOT.resolve():
    raise RuntimeError("Could not safely import the original Phase 1 configuration.")
phase1_config = phase1.cfg
