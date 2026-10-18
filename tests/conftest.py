"""Isolate the test suite from the developer's own llmux state.

`tui.common.paths` resolves the data root at import time, and a checkout is
its own data root — so without this, tests would read the local
`profiles.yaml`, `.env.common`, `config/` and `.runtime/`. Point the data root
at a throwaway directory shaped like a fresh clone before any `tui` import.

The variable is set only while `PATHS` resolves: subprocess-based tests build
their own sandbox via `LLMUX_ROOT`, which an inherited `LLMUX_DATA_DIR` would
override.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[1]

_data_root = Path(tempfile.mkdtemp(prefix="llmux-test-data-"))
atexit.register(shutil.rmtree, _data_root, ignore_errors=True)
for backend in ("vllm", "llamacpp"):
    example = _CHECKOUT / "config" / backend / "example.yaml"
    if example.is_file():
        (_data_root / "config" / backend).mkdir(parents=True)
        shutil.copy2(example, _data_root / "config" / backend / "example.yaml")

os.environ.pop("LLMUX_ROOT", None)
os.environ["LLMUX_DATA_DIR"] = str(_data_root)
try:
    import tui.common.paths  # noqa: E402,F401  (resolves PATHS once)
finally:
    os.environ.pop("LLMUX_DATA_DIR", None)
