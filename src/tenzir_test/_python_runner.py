from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Dict

from . import fixtures as _fixtures
from .fixtures import acquire_fixture as _acquire_fixture


_EXPORTED_HELPERS: dict[str, Any] = {
    "fixtures": _fixtures.fixtures_api,
    "fixtures_module": _fixtures,
    "fixtures_selection": _fixtures.fixtures,
    "activate": _fixtures.activate,
    "require": _fixtures.require,
    "has": _fixtures.has,
    "FixtureContext": _fixtures.FixtureContext,
    "FixtureHandle": _fixtures.FixtureHandle,
    "FixtureSelection": _fixtures.FixtureSelection,
    "FixtureController": _fixtures.FixtureController,
    "Executor": _fixtures.Executor,
}

_EXPORTED_HELPERS["acquire_fixture"] = _acquire_fixture


def _build_init_globals() -> Dict[str, Any]:
    init_globals: Dict[str, Any] = {}
    init_globals.update(_EXPORTED_HELPERS)
    return init_globals


def _run_script(script_path: Path, args: list[str]) -> None:
    sys.argv = [str(script_path), *args]
    init_globals = _build_init_globals()
    runpy.run_path(str(script_path), run_name="__main__", init_globals=init_globals)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv if argv is None else argv)
    if len(args) < 2:
        raise SystemExit("usage: python -m tenzir_test._python_runner <path> [args...]")

    script_arg = args[1]
    script_path = Path(script_arg)
    if not script_path.exists():
        raise SystemExit(f"python runner could not find test file: {script_arg}")

    extra_args = args[2:]
    _run_script(script_path, extra_args)


if __name__ == "__main__":
    main()
