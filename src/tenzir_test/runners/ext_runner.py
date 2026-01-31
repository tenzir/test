from __future__ import annotations

from pathlib import Path

from ._utils import get_run_module
from .runner import Runner

# Extensions reserved by the framework and not available for custom runners.
# These extensions have special meaning in the test harness:
#   - .txt: baseline output file for comparison
#   - .input: inline input data exposed via TENZIR_INPUT env var
#   - .stdin: stdin content fed to test process via stdin_data parameter
#
# Custom runner authors who need stdin support should use get_stdin_content()
# and pass the result to run_subprocess(stdin_data=...). See the ShellRunner
# and CustomPythonFixture implementations for reference.
RESERVED_EXTENSIONS = frozenset({"input", "stdin", "txt"})


class ExtRunner(Runner):
    def __init__(self, *, name: str, ext: str) -> None:
        if ext in RESERVED_EXTENSIONS:
            raise ValueError(f"runner '{name}' uses reserved extension '.{ext}'")
        super().__init__(name=name)
        self._ext = ext

    def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
        return self.collect_with_ext(path, self._ext)

    def purge(self) -> None:
        run_mod = get_run_module()
        purge_base = run_mod.ROOT / self._name
        if not purge_base.exists():
            return
        for p in purge_base.rglob("*"):
            if p.is_dir() or p.suffix == f".{self._ext}":
                continue
            p.unlink()


__all__ = ["ExtRunner"]
