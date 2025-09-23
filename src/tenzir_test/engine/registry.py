from __future__ import annotations

from collections.abc import Iterable

from tenzir_test import runners as runner_module
from tenzir_test.runners import Runner


def iter_runners() -> Iterable[Runner]:
    return runner_module.iter_runners()


def get_runner(name: str) -> Runner:
    return runner_module.runner_map()[name]


def has_runner(name: str) -> bool:
    return name in runner_module.runner_map()


def allowed_extensions() -> set[str]:
    return runner_module.allowed_extensions()


__all__ = [
    "iter_runners",
    "get_runner",
    "has_runner",
    "allowed_extensions",
]
