from __future__ import annotations

import inspect
import logging
import os
import subprocess
import sys
import time
from contextlib import ExitStack, AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Protocol, Sequence


_FIXTURES_ENV = "TENZIR_TEST_FIXTURES"


@dataclass(frozen=True)
class FixtureContext:
    """Describe the invocation context available to fixture factories."""

    test: Path
    config: dict[str, Any]
    coverage: bool
    env: dict[str, str]
    config_args: Sequence[str]
    tenzir_binary: str | None
    tenzir_node_binary: str | None


_CONTEXT: ContextVar[FixtureContext | None] = ContextVar(
    "tenzir_test_fixture_context", default=None
)

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self) -> None:
        self.binary: str = os.environ["TENZIR_NODE_CLIENT_BINARY"]
        self.endpoint: str | None = os.environ.get("TENZIR_NODE_CLIENT_ENDPOINT")
        timeout_raw = os.environ.get("TENZIR_NODE_CLIENT_TIMEOUT")
        self.remaining_timeout: float = float(timeout_raw) if timeout_raw is not None else 0.0

    def run(
        self, source: str, desired_timeout: float | None = None, mirror: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        cmd = [
            self.binary,
            "--bare-mode",
            "--console-verbosity=warning",
            "--multi",
        ]
        if self.endpoint is not None:
            cmd.append(f"--endpoint={self.endpoint}")
        cmd.append(source)
        start = time.process_time()
        requested_timeout = (
            desired_timeout if desired_timeout is not None else self.remaining_timeout
        )
        timeout = min(self.remaining_timeout, requested_timeout)
        res = subprocess.run(cmd, timeout=timeout, capture_output=True)
        end = time.process_time()
        used_time = end - start
        self.remaining_timeout = max(0, self.remaining_timeout - used_time)
        if mirror:
            if res.stdout:
                print(res.stdout.decode())
            if res.stderr:
                print(res.stderr.decode(), file=sys.stderr)
        return res


def _parse_fixture_env(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    parts = [part.strip() for part in raw.split(",")]
    return frozenset(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class FixtureSelection:
    names: frozenset[str]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __bool__(self) -> bool:
        return bool(self.names)

    def __contains__(self, item: str) -> bool:
        return item in self.names

    def has(self, name: str) -> bool:
        return name in self.names

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.names]
        if missing:
            missing_list = ", ".join(sorted(missing))
            available = ", ".join(sorted(self.names)) or "<none>"
            raise RuntimeError(
                f"Missing required fixture(s): {missing_list} (available: {available})"
            )

    def any_of(self, names: Iterable[str]) -> bool:
        for name in names:
            if name in self.names:
                return True
        return False

    def as_tuple(self) -> tuple[str, ...]:
        return tuple(sorted(self.names))


def requested() -> FixtureSelection:
    """Return the fixture selection encoded in the TENZIR_TEST_FIXTURES env var."""

    return FixtureSelection(_parse_fixture_env(os.environ.get(_FIXTURES_ENV)))


def has(name: str) -> bool:
    """Check whether the given fixture was requested."""

    return name in requested()


def require(*names: str) -> None:
    """Assert that all requested fixtures are present, raising RuntimeError otherwise."""

    requested().require(*names)


def push_context(context: FixtureContext) -> Token:
    """Install the given context for the duration of fixture activation."""

    return _CONTEXT.set(context)


def pop_context(token: Token) -> None:
    """Restore the previous fixture context."""

    _CONTEXT.reset(token)


def current_context() -> FixtureContext | None:
    """Return the active fixture context, if any."""

    return _CONTEXT.get()


FixtureFactory = Callable[[], ContextManager[dict[str, str] | None]]


@dataclass(slots=True)
class FixtureHandle:
    """Container describing a fixture environment and optional teardown hook."""

    env: dict[str, str] | None = None
    teardown: Callable[[], None] | None = None


class _FactoryCallable(Protocol):
    def __call__(
        self,
    ) -> (
        ContextManager[dict[str, str] | None]
        | FixtureHandle
        | dict[str, str]
        | tuple[dict[str, str] | None, Callable[[], None] | None]
        | None
    ): ...


_FACTORIES: dict[str, FixtureFactory] = {}


def _normalize_factory(factory: _FactoryCallable) -> FixtureFactory:
    def _as_context_manager() -> ContextManager[dict[str, str] | None]:
        result = factory()
        if isinstance(result, AbstractContextManager):
            return result
        if isinstance(result, FixtureHandle):
            env_dict: dict[str, str] = result.env or {}

            @contextmanager
            def _ctx() -> Iterator[dict[str, str] | None]:
                try:
                    yield env_dict
                finally:
                    if result.teardown:
                        result.teardown()

            return _ctx()
        if isinstance(result, tuple) and len(result) == 2:
            raw_env, teardown = result
            env_dict = raw_env or {}

            @contextmanager
            def _ctx() -> Iterator[dict[str, str] | None]:
                try:
                    yield env_dict
                finally:
                    if callable(teardown):
                        teardown()

            return _ctx()
        if result is None:

            @contextmanager
            def _ctx_none() -> Iterator[dict[str, str] | None]:
                yield {}

            return _ctx_none()
        if isinstance(result, dict):

            @contextmanager
            def _ctx_dict() -> Iterator[dict[str, str] | None]:
                yield result

            return _ctx_dict()
        raise TypeError(
            "fixture factory must return a context manager, FixtureHandle, dict,"
            " tuple[env, teardown], or None"
        )

    return _as_context_manager


def _infer_name(func: Callable[..., object], explicit: str | None) -> str:
    if explicit:
        return explicit
    code_obj = getattr(func, "__code__", None)
    if code_obj is not None and hasattr(code_obj, "co_filename"):
        file = Path(code_obj.co_filename)
        return file.stem
    name_attr = getattr(func, "__name__", None)
    if isinstance(name_attr, str):
        return name_attr
    raise ValueError("Unable to infer fixture name; please provide one explicitly")


def register(name: str | None, factory: _FactoryCallable, *, replace: bool = False) -> None:
    resolved_name = _infer_name(factory, name)
    if resolved_name in _FACTORIES and not replace:
        raise ValueError(f"fixture '{resolved_name}' already registered")
    _FACTORIES[resolved_name] = _normalize_factory(factory)


def fixture(
    func: _FactoryCallable | None = None,
    *,
    name: str | None = None,
    replace: bool = False,
    log_teardown: bool = False,
) -> Callable[[_FactoryCallable], _FactoryCallable] | _FactoryCallable:
    """Decorator registering a fixture factory.

    ``@fixture`` accepts generator functions, context managers, or callables that
    return any of the supported fixture factory types. When used on a generator
    function, the decorator implicitly wraps it with :func:`contextlib.contextmanager`
    so authors can ``yield`` environments directly.
    """

    def _decorator(inner: _FactoryCallable) -> _FactoryCallable:
        resolved_name = _infer_name(inner, name)
        candidate: _FactoryCallable
        if inspect.isgeneratorfunction(inner):
            candidate = contextmanager(inner)
        else:
            candidate = inner

        register(resolved_name, candidate, replace=replace)

        if log_teardown:
            registered = _FACTORIES.get(resolved_name)
            if registered is not None:
                setattr(registered, "tenzir_log_teardown", True)

        return inner

    if func is not None:
        return _decorator(func)

    return _decorator


@contextmanager
def activate(names: Iterable[str]) -> Iterator[dict[str, str]]:
    stack = ExitStack()
    combined: dict[str, str] = {}

    def _wrap_factory(
        factory: FixtureFactory,
        *,
        name: str,
        force_teardown_log: bool,
    ) -> ContextManager[dict[str, str] | None]:
        @contextmanager
        def _logged_context() -> Iterator[dict[str, str] | None]:
            logger.info("activating fixture '%s'", name)
            should_log_teardown = force_teardown_log
            with factory() as env:
                keys: tuple[str, ...] = tuple()
                if env:
                    keys = tuple(sorted(env.keys()))
                    logger.info(
                        "fixture '%s' provided context keys: %s",
                        name,
                        ", ".join(keys),
                    )
                    should_log_teardown = True
                try:
                    yield env
                finally:
                    if should_log_teardown:
                        logger.info("tearing down fixture '%s'", name)

        return _logged_context()

    try:
        for name in names:
            factory = _FACTORIES.get(name)
            if not factory:
                logger.debug("requested fixture '%s' has no registered factory", name)
                continue
            force_teardown_log = bool(getattr(factory, "tenzir_log_teardown", False))
            env: dict[str, str] | None = stack.enter_context(
                _wrap_factory(factory, name=name, force_teardown_log=force_teardown_log)
            )
            if env:
                combined.update(env)
        yield combined
    finally:
        stack.close()


# Import built-in fixtures so they self-register on package import.
from . import node  # noqa: F401,E402
