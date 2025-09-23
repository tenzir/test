from __future__ import annotations

from .diff_runner import DiffRunner


class InstantiationRunner(DiffRunner):
    def __init__(self) -> None:
        super().__init__(a="--dump-ir", b="--dump-inst-ir", name="instantiation")


__all__ = ["InstantiationRunner"]
