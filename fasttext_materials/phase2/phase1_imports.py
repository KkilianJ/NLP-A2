"""Import the untouched Phase 1 code and direct temporary caches to Phase 2."""
import os
import sys

import settings

os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(settings.CACHE_DIR / "matplotlib"))
sys.path.insert(0, str(settings.ASSIGNMENT_ROOT))
_previous_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import config as phase1_config  # noqa: E402
    import run_phase1 as phase1  # noqa: E402
finally:
    sys.dont_write_bytecode = _previous_bytecode_setting
