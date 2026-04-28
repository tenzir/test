"""Project hook registration and invocation for tenzir-test."""

from __future__ import annotations

import contextvars
import dataclasses
import os
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast

HookEvent = Literal[
    "startup",
    "shutdown",
    "project_start",
    "project_finish",
    "test_start",
    "test_finish",
    "test_failure",
]

T = TypeVar("T")


@dataclasses.dataclass(slots=True)
class HookSet:
    startup: list[Callable[["StartupContext"], None]] = dataclasses.field(default_factory=list)
    shutdown: list[Callable[["ShutdownContext"], None]] = dataclasses.field(default_factory=list)
    project_start: list[Callable[["ProjectStartContext"], None]] = dataclasses.field(
        default_factory=list
    )
    project_finish: list[Callable[["ProjectFinishContext"], None]] = dataclasses.field(
        default_factory=list
    )
    test_start: list[Callable[["TestStartContext"], None]] = dataclasses.field(default_factory=list)
    test_finish: list[Callable[["TestFinishContext"], None]] = dataclasses.field(
        default_factory=list
    )
    test_failure: list[Callable[["TestFailureContext"], None]] = dataclasses.field(
        default_factory=list
    )

    def is_empty(self) -> bool:
        return not any(getattr(self, event) for event in HOOK_EVENTS)


HOOK_EVENTS: tuple[HookEvent, ...] = (
    "startup",
    "shutdown",
    "project_start",
    "project_finish",
    "test_start",
    "test_finish",
    "test_failure",
)

_loading_hooks: contextvars.ContextVar[HookSet | None] = contextvars.ContextVar(
    "tenzir_test_loading_hooks", default=None
)


class HookEnvironment(MutableMapping[str, str]):
    """Mutable hook environment with PATH managed through a separate list."""

    def __init__(self, env: MutableMapping[str, str], path: list[str]) -> None:
        self._env = env
        self._path = path

    def __getitem__(self, key: str) -> str:
        if key == "PATH":
            return os.pathsep.join(self._path)
        return self._env[key]

    def __setitem__(self, key: str, value: str) -> None:
        if key == "PATH":
            raise RuntimeError("modify ctx.path instead of ctx.env['PATH']")
        self._env[key] = value

    def __delitem__(self, key: str) -> None:
        if key == "PATH":
            raise RuntimeError("modify ctx.path instead of ctx.env['PATH']")
        del self._env[key]

    def __iter__(self) -> Iterator[str]:
        yielded_path = False
        for key in self._env:
            if key == "PATH":
                yielded_path = True
            yield key
        if not yielded_path and self._path:
            yield "PATH"

    def __len__(self) -> int:
        return len(self._env) + (0 if "PATH" in self._env or not self._path else 1)

    def clear(self) -> None:
        self._env.clear()
        self._path.clear()


def _register(event: HookEvent, func: T) -> T:
    current = _loading_hooks.get()
    if current is None:
        raise RuntimeError(f"@hooks.{event} can only be used while tenzir-test loads project hooks")
    getattr(current, event).append(func)
    return func


def startup(func: Callable[["StartupContext"], None]) -> Callable[["StartupContext"], None]:
    return _register("startup", func)


def shutdown(func: Callable[["ShutdownContext"], None]) -> Callable[["ShutdownContext"], None]:
    return _register("shutdown", func)


def project_start(
    func: Callable[["ProjectStartContext"], None],
) -> Callable[["ProjectStartContext"], None]:
    return _register("project_start", func)


def project_finish(
    func: Callable[["ProjectFinishContext"], None],
) -> Callable[["ProjectFinishContext"], None]:
    return _register("project_finish", func)


def test_start(func: Callable[["TestStartContext"], None]) -> Callable[["TestStartContext"], None]:
    return _register("test_start", func)


def test_finish(
    func: Callable[["TestFinishContext"], None],
) -> Callable[["TestFinishContext"], None]:
    return _register("test_finish", func)


def test_failure(
    func: Callable[["TestFailureContext"], None],
) -> Callable[["TestFailureContext"], None]:
    return _register("test_failure", func)


class loading(HookSet):
    """Context manager that makes decorators register in this hook set."""

    _token: contextvars.Token[HookSet | None]

    def __enter__(self) -> HookSet:
        self._token = _loading_hooks.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _loading_hooks.reset(self._token)


@dataclasses.dataclass(frozen=True, slots=True)
class SummaryView:
    failed: int
    total: int
    skipped: int


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectView:
    root: Path
    kind: Literal["root", "satellite"]


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectResultView:
    project: ProjectView
    summary: SummaryView
    queue_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class FixtureSpecView:
    name: str
    options: object | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteView:
    name: str
    directory: Path


@dataclasses.dataclass(slots=True)
class StartupContext:
    root: Path
    env: MutableMapping[str, str]
    path: list[str]
    debug: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ShutdownContext:
    root: Path
    exit_code: int
    interrupted: bool
    summary: SummaryView
    project_results: tuple[ProjectResultView, ...]
    debug: bool


@dataclasses.dataclass(slots=True)
class ProjectStartContext:
    root: Path
    project: ProjectView
    previous_project: ProjectView | None
    kind: Literal["root", "satellite"]
    env: MutableMapping[str, str]
    path: list[str]
    debug: bool
    update: bool
    coverage: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectFinishContext:
    root: Path
    project: ProjectView
    next_project: ProjectView | None
    kind: Literal["root", "satellite"]
    summary: SummaryView
    interrupted: bool
    debug: bool


@dataclasses.dataclass(frozen=True, slots=True)
class TestStartContext:
    root: Path
    project: ProjectView
    test: Path
    runner: str
    fixtures: tuple[FixtureSpecView, ...]
    suite: SuiteView | None
    update: bool
    coverage: bool
    attempt_limit: int


@dataclasses.dataclass(frozen=True, slots=True)
class TestFinishContext:
    root: Path
    project: ProjectView
    test: Path
    runner: str
    outcome: Literal["passed", "failed", "skipped"]
    reason: str | None
    attempts: int
    duration: float
    fixtures: tuple[FixtureSpecView, ...]
    suite: SuiteView | None
    tmp_dir: Path | None
    update: bool
    coverage: bool


@dataclasses.dataclass(frozen=True, slots=True)
class TestFailureContext:
    root: Path
    project: ProjectView
    test: Path
    runner: str
    reason: str | None
    attempts: int
    duration: float
    fixtures: tuple[FixtureSpecView, ...]
    suite: SuiteView | None
    tmp_dir: Path | None
    update: bool
    coverage: bool


class HookInvocationError(RuntimeError):
    pass


_TEARDOWN_EVENTS: frozenset[HookEvent] = frozenset({"shutdown", "project_finish", "test_finish"})


def invoke(
    hook_sets: Sequence[HookSet],
    event: HookEvent,
    context: object,
    *,
    reverse: bool = False,
    project_root: Path | None = None,
    test_path: Path | None = None,
    debug: bool = False,
) -> None:
    ordered_sets = tuple(reversed(hook_sets)) if reverse else tuple(hook_sets)
    errors: list[HookInvocationError] = []
    for hook_set in ordered_sets:
        funcs = tuple(getattr(hook_set, event))
        ordered_funcs = tuple(reversed(funcs)) if reverse else funcs
        for func in ordered_funcs:
            if debug:
                name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
                module = getattr(func, "__module__", "<unknown>")
                print(f"debug: invoking hook {event} {module}.{name}")
            try:
                cast(Callable[[object], None], func)(context)
            except BaseException as exc:
                name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
                module = getattr(func, "__module__", "<unknown>")
                parts = [f"hook {event} {name} failed"]
                if test_path is not None:
                    parts.append(f"for {test_path}")
                if project_root is not None:
                    parts.append(f"in {project_root}")
                parts.append(f"({module}): {exc}")
                error = HookInvocationError(" ".join(parts))
                if event in _TEARDOWN_EVENTS:
                    errors.append(error)
                    continue
                raise error from exc
    if errors:
        first = errors[0]
        for error in errors[1:]:
            first.add_note(str(error))
        raise first
