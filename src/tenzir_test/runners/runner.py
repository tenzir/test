from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Runner(ABC):
    def __init__(self, *, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def collect_with_ext(self, path: Path, ext: str) -> set[tuple[Runner, Path]]:
        todo: set[tuple[Runner, Path]] = set()
        if path.is_file():
            if path.suffix == f".{ext}":
                todo.add((self, path))
            return todo
        for test in path.glob(f"**/*.{ext}"):
            todo.add((self, test))
        return todo

    @abstractmethod
    def collect_tests(self, path: Path) -> set[tuple[Runner, Path]]:
        raise NotImplementedError

    @abstractmethod
    def purge(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        raise NotImplementedError


__all__ = ["Runner"]
