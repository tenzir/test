from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from .runner import Runner
from .ext_runner import ExtRunner
from .tql_runner import TqlRunner
from .diff_runner import DiffRunner
from .shell_runner import ShellRunner
from .custom_python_fixture_runner import CustomPythonFixture
from .tenzir_runner import TenzirRunner

_REGISTERED: dict[str, Runner] = {}
_ALIASES: dict[str, str] = {}
RUNNERS: list[Runner] = []
RUNNERS_BY_NAME: dict[str, Runner] = {}


def _refresh_exports() -> None:
    global RUNNERS, RUNNERS_BY_NAME
    RUNNERS = list(_REGISTERED.values())
    mapping: dict[str, Runner] = dict(_REGISTERED)
    for alias, target in _ALIASES.items():
        runner = mapping.get(target)
        if runner is None:
            continue
        mapping[alias] = runner
    RUNNERS_BY_NAME = mapping


def _set_alias(alias: str, target: str, *, replace: bool) -> None:
    if target not in _REGISTERED:
        raise ValueError(f"runner '{target}' is not registered")
    if alias in _REGISTERED and alias != target:
        raise ValueError(f"cannot alias '{alias}': name already registered")
    if not replace and alias in _ALIASES and _ALIASES[alias] != target:
        raise ValueError(f"alias '{alias}' already targets '{_ALIASES[alias]}'")
    _ALIASES[alias] = target


def register(
    runner: Runner,
    *,
    replace: bool = False,
    aliases: Sequence[str] | None = None,
) -> Runner:
    name = runner.name
    if not replace and name in _REGISTERED and _REGISTERED[name] is not runner:
        raise ValueError(f"runner '{name}' already registered")
    _REGISTERED[name] = runner
    if aliases:
        for alias in aliases:
            _set_alias(alias, name, replace=replace)
    _refresh_exports()
    return runner


def unregister(name: str) -> Runner:
    try:
        runner = _REGISTERED.pop(name)
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"runner '{name}' is not registered") from exc
    to_remove = [alias for alias, target in _ALIASES.items() if target == name]
    for alias in to_remove:
        _ALIASES.pop(alias, None)
    _refresh_exports()
    return runner


def register_alias(alias: str, target: str, *, replace: bool = False) -> None:
    _set_alias(alias, target, replace=replace)
    _refresh_exports()


RunnerFactory = Callable[[], Runner]


def startup(
    *,
    replace: bool = False,
    aliases: Sequence[str] | None = None,
) -> Callable[[RunnerFactory], RunnerFactory]:
    def _decorator(factory: RunnerFactory) -> RunnerFactory:
        runner = factory()
        if not isinstance(runner, Runner):
            raise TypeError("runner factory must return a Runner instance")
        register(runner, replace=replace, aliases=aliases)
        return factory

    return _decorator


def iter_runners() -> tuple[Runner, ...]:
    return tuple(RUNNERS)


def runner_map(*, include_aliases: bool = True) -> dict[str, Runner]:
    if include_aliases:
        return dict(RUNNERS_BY_NAME)
    return dict(_REGISTERED)


def runner_names() -> set[str]:
    return set(RUNNERS_BY_NAME.keys())


def allowed_extensions() -> set[str]:
    extensions: set[str] = set()
    for runner in RUNNERS:
        ext = getattr(runner, "_ext", None)
        if isinstance(ext, str):
            extensions.add(ext)
    return extensions


def get_runner_for_test(test_path: Path) -> Runner:
    from tenzir_test import run as run_module

    config = run_module.parse_test_config(test_path)
    runner_value = config.get("runner")
    if not isinstance(runner_value, str):
        raise ValueError("Runner 'runner' must be a string")
    runner_name = runner_value
    if runner_name in RUNNERS_BY_NAME:
        return RUNNERS_BY_NAME[runner_name]
    raise ValueError(f"Runner '{runner_name}' not found - this is a bug")


register(ShellRunner())
register(CustomPythonFixture())
register(TenzirRunner())

_refresh_exports()

__all__ = [
    "Runner",
    "ExtRunner",
    "TqlRunner",
    "ShellRunner",
    "CustomPythonFixture",
    "TenzirRunner",
    "DiffRunner",
    "RUNNERS",
    "RUNNERS_BY_NAME",
    "register",
    "unregister",
    "register_alias",
    "startup",
    "iter_runners",
    "runner_map",
    "runner_names",
    "allowed_extensions",
    "get_runner_for_test",
]
