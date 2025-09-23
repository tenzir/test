from __future__ import annotations

import os
from types import ModuleType


def get_run_module() -> ModuleType:
    from tenzir_test import run as run_module

    return run_module


def resolve_run_module_dir(run_mod: ModuleType) -> str:
    module_path = getattr(run_mod, "__file__", None)
    if not isinstance(module_path, str):
        raise RuntimeError("tenzir_test.run module path is not available")
    return os.path.dirname(os.path.realpath(module_path))
