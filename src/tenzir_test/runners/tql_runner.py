from __future__ import annotations

from .ext_runner import ExtRunner


class TqlRunner(ExtRunner):
    output_ext: str = "txt"

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, ext="tql")


__all__ = ["TqlRunner"]
