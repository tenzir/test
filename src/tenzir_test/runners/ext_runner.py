from __future__ import annotations

from pathlib import Path

from ._utils import get_run_module
from .runner import Runner


class ExtRunner(Runner):
    def __init__(self, *, name: str, ext: str) -> None:
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
