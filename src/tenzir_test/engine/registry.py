from __future__ import annotations

from collections.abc import Iterable

from tenzir_test import run


def iter_runners() -> Iterable[run.Runner]:
    return tuple(run.RUNNERS)


def get_runner(prefix: str) -> run.Runner:
    return run.runners[prefix]


def has_runner(prefix: str) -> bool:
    return prefix in run.runners


def allowed_extensions() -> set[str]:
    return run.get_allowed_extensions()


__all__ = [
    "iter_runners",
    "get_runner",
    "has_runner",
    "allowed_extensions",
]
