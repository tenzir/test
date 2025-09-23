from __future__ import annotations

from .diff_runner import DiffRunner


class OptRunner(DiffRunner):
    def __init__(self) -> None:
        super().__init__(a="--dump-inst-ir", b="--dump-opt-ir", name="opt")


__all__ = ["OptRunner"]
