from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from abc import ABC, abstractmethod
from pathlib import Path


@dataclasses.dataclass(frozen=True, slots=True)
class RequirementCheckResult:
    """Outcome of evaluating suite requirements for a runner."""

    unsupported_keys: tuple[str, ...] = tuple()
    missing_values: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)


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

    def check_requirements(
        self,
        requirements: Mapping[str, Sequence[str]],
        *,
        env: Mapping[str, str],
        config_args: Sequence[str],
    ) -> RequirementCheckResult:
        """Return unsupported and missing requirements for this runner.

        Base runners do not expose requirement probes and therefore report all
        non-empty requirement categories as unsupported.
        """
        unsupported = tuple(
            sorted(key for key, values in requirements.items() if values)
        )
        return RequirementCheckResult(unsupported_keys=unsupported)


__all__ = ["Runner", "RequirementCheckResult"]
