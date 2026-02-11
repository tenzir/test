from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import types
import typing
from contextlib import ExitStack, AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    Literal,
)


_FIXTURES_ENV = "TENZIR_TEST_FIXTURES"
_HOOKS_ATTR = "__tenzir_fixture_hooks__"

_TMP_DIR_CLEANUP: dict[Path, Callable[[], None]] = {}
_TMP_DIR_CLEANUP_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    """Pair a fixture name with an optional options dict."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.name, json.dumps(self.options, sort_keys=True)))

    def __str__(self) -> str:
        if not self.options:
            return self.name
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.options.items()))
        return f"{self.name}({items})"


_OPTIONS_CLASSES: dict[str, type] = {}


def _unwrap_optional(tp: Any) -> Any:
    """Return inner type for Optional[T] or T | None annotations."""
    origin = typing.get_origin(tp)
    if origin not in (typing.Union, types.UnionType):
        return tp
    non_none_args = [arg for arg in typing.get_args(tp) if arg is not type(None)]
    return non_none_args[0] if len(non_none_args) == 1 else tp


def _instantiate_options(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively instantiate nested dataclass options from mapping data."""
    try:
        hints = typing.get_type_hints(cls)
    except NameError:
        # Fall back to non-recursive construction when forward refs are unavailable.
        hints = {}
    processed: dict[str, Any] = {}
    for key, value in data.items():
        raw_type = hints.get(key)
        field_type = _unwrap_optional(raw_type) if raw_type is not None else None
        if (
            field_type is not None
            and isinstance(field_type, type)
            and dataclasses.is_dataclass(field_type)
            and isinstance(value, Mapping)
        ):
            processed[key] = _instantiate_options(field_type, value)
        else:
            processed[key] = value
    return cls(**processed)


def get_options_class(name: str) -> type | None:
    """Return the registered options dataclass for the named fixture, if any."""
    return _OPTIONS_CLASSES.get(name)


@dataclass(frozen=True)
class FixtureContext:
    """Describe the invocation context available to fixture factories."""

    test: Path
    config: dict[str, Any]
    coverage: bool
    env: dict[str, str]
    config_args: Sequence[str]
    tenzir_binary: tuple[str, ...] | None
    tenzir_node_binary: tuple[str, ...] | None
    fixture_options: Mapping[str, Any] = field(default_factory=dict)


_CONTEXT: ContextVar[FixtureContext | None] = ContextVar(
    "tenzir_test_fixture_context", default=None
)

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        source = env or os.environ
        try:
            self.binary: tuple[str, ...] = tuple(shlex.split(source["TENZIR_NODE_CLIENT_BINARY"]))
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("TENZIR_NODE_CLIENT_BINARY is not configured") from exc
        self.endpoint: str | None = source.get("TENZIR_NODE_CLIENT_ENDPOINT")
        timeout_raw = source.get("TENZIR_NODE_CLIENT_TIMEOUT")
        self.remaining_timeout: float = float(timeout_raw) if timeout_raw is not None else 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Executor":
        return cls(env=env)

    def run(
        self, source: str, desired_timeout: float | None = None, mirror: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        cmd = [
            *self.binary,
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
        if logger.isEnabledFor(logging.DEBUG):
            command_line = shlex.join(str(part) for part in cmd)
            logger.debug("executing fixture command: %s", command_line)
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

    def __getattr__(self, item: str) -> bool:
        if item in self.names:
            return True
        raise AttributeError(
            f"fixture '{item}' was not requested; available fixtures: "
            f"{', '.join(sorted(self.names)) or '<none>'}"
        )

    def __dir__(self) -> list[str]:
        base = set(super().__dir__())
        base.update(self.names)
        return sorted(base)

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


def fixtures() -> FixtureSelection:
    """Return the current fixture selection."""

    return FixtureSelection(_parse_fixture_env(os.environ.get(_FIXTURES_ENV)))


def has(name: str) -> bool:
    """Check whether the given fixture was requested."""

    return name in fixtures()


def require(*names: str) -> None:
    """Assert that all requested fixtures are present, raising RuntimeError otherwise."""

    fixtures().require(*names)


class FixturesAccessor:
    def __call__(self) -> FixtureSelection:
        return fixtures()

    def __getattr__(self, item: str) -> Any:
        module = sys.modules[__name__]
        return getattr(module, item)

    def __dir__(self) -> list[str]:
        module = sys.modules[__name__]
        base = set(dir(module))
        try:
            base.update(fixtures().names)
        except Exception:  # pragma: no cover - defensive
            pass
        return sorted(base)


fixtures_api = FixturesAccessor()


class FixtureController:
    """Imperative controller for manually driving a fixture lifecycle."""

    def __init__(self, name: str, factory: FixtureFactory) -> None:
        self._name = name
        self._factory = factory
        self._force_teardown_log = bool(getattr(factory, "tenzir_log_teardown", False))
        self._state: tuple[ContextManager[dict[str, str] | None], bool] | None = None
        self.env: dict[str, str] = {}
        self._hooks: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> "FixtureController":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> Literal[False]:
        self.stop()
        return False

    @property
    def is_running(self) -> bool:
        return self._state is not None

    def start(self) -> dict[str, str]:
        if self._state is not None:
            raise RuntimeError(f"fixture '{self._name}' is already running")

        context = self._factory()
        logger.info("activating fixture '%s'", self._name)
        should_log_teardown = self._force_teardown_log

        env = context.__enter__()
        env_dict = env or {}
        if env_dict:
            keys = tuple(sorted(env_dict.keys()))
            logger.info("fixture '%s' provided context keys:", self._name)
            for key in keys:
                logger.info("  - %s", key)
            should_log_teardown = True

        hooks = getattr(context, _HOOKS_ATTR, {}) or {}
        self._hooks = {
            name: self._wrap_hook(name, hook) for name, hook in hooks.items() if callable(hook)
        }

        self.env = env_dict
        self._state = (context, should_log_teardown)
        return self.env

    def stop(self) -> None:
        if self._state is None:
            return
        context, should_log_teardown = self._state
        self._state = None
        try:
            context.__exit__(None, None, None)
        finally:
            if should_log_teardown:
                logger.info("tearing down fixture '%s'", self._name)
            self.env = {}
            self._hooks.clear()

    def restart(self) -> dict[str, str]:
        self.stop()
        return self.start()

    def _wrap_hook(self, name: str, hook: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(hook)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            if self._state is None:
                raise RuntimeError(
                    f"cannot call '{name}' on fixture '{self._name}' because it is not running"
                )
            return hook(*args, **kwargs)

        return _inner

    def __getattr__(self, item: str) -> Any:
        hooks = self._hooks
        if item in hooks:
            return hooks[item]
        raise AttributeError(f"fixture controller has no attribute '{item}'")

    def __dir__(self) -> list[str]:
        base = set(super().__dir__())
        base.update(self._hooks.keys())
        return sorted(base)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        status = "running" if self.is_running else "stopped"
        return f"FixtureController(name={self._name!r}, status={status})"


def acquire_fixture(name: str) -> FixtureController:
    """Return a controller for manually driving the named fixture."""

    factory = _FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_FACTORIES.keys())) or "<none>"
        raise ValueError(f"fixture '{name}' is not registered (available: {available})")
    return FixtureController(name, factory)


def push_context(context: FixtureContext) -> Token:
    """Install the given context for the duration of fixture activation."""

    return _CONTEXT.set(context)


def pop_context(token: Token) -> None:
    """Restore the previous fixture context."""

    _CONTEXT.reset(token)


def register_tmp_dir_cleanup(path: str | os.PathLike[str], callback: Callable[[], None]) -> None:
    """Register a cleanup callback for a temporary directory."""

    normalized = Path(path).resolve()
    with _TMP_DIR_CLEANUP_LOCK:
        _TMP_DIR_CLEANUP[normalized] = callback


def unregister_tmp_dir_cleanup(path: str | os.PathLike[str]) -> None:
    """Remove a previously registered cleanup callback, if any."""

    normalized = Path(path).resolve()
    with _TMP_DIR_CLEANUP_LOCK:
        _TMP_DIR_CLEANUP.pop(normalized, None)


def invoke_tmp_dir_cleanup(path: str | os.PathLike[str]) -> None:
    """Execute and discard the cleanup callback registered for `path`, if present."""

    normalized = Path(path).resolve()
    with _TMP_DIR_CLEANUP_LOCK:
        targets = [
            _TMP_DIR_CLEANUP.pop(candidate)
            for candidate in tuple(_TMP_DIR_CLEANUP)
            if candidate == normalized or candidate.is_relative_to(normalized)
        ]
    for callback in targets:
        callback()


def current_context() -> FixtureContext | None:
    """Return the active fixture context, if any."""

    return _CONTEXT.get()


def current_options(name: str) -> Any:
    """Return options for the named fixture from the active context.

    Returns the typed dataclass instance if the fixture declared one,
    otherwise the raw dict. Returns an empty dict if no context is active
    or no options were provided and no options class is registered.
    """
    ctx = _CONTEXT.get()
    if ctx is None:
        return {}
    return ctx.fixture_options.get(name, {})


FixtureFactory = Callable[[], ContextManager[dict[str, str] | None]]


@dataclass(slots=True)
class FixtureHandle:
    """Container describing a fixture environment and optional teardown hook."""

    env: dict[str, str] | None = None
    teardown: Callable[[], None] | None = None
    hooks: Mapping[str, Callable[..., Any]] | None = None


class FixtureUnavailable(Exception):
    """Raised by a fixture when it cannot provide its service.

    When a fixture raises this during initialization (before yielding)
    and the suite has ``skip: {on: fixture-unavailable}`` in its
    ``test.yaml``, all tests in the suite are marked as skipped.

    Without the opt-in config, the exception propagates normally and
    causes a test failure.

    Example::

        @fixture()
        def my_fixture() -> Iterator[dict[str, str]]:
            if not shutil.which("docker"):
                raise FixtureUnavailable(
                    "container runtime (docker) required but not found"
                )
            # ... normal setup ...
    """


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


@dataclass(slots=True)
class _SuiteScope:
    fixtures: tuple[FixtureSpec, ...]
    stack: ExitStack
    env: dict[str, str]
    depth: int = 0


_SUITE_SCOPE: ContextVar[_SuiteScope | None] = ContextVar(
    "tenzir_test_fixture_suite_scope", default=None
)


def _attach_hooks(
    manager: ContextManager[dict[str, str] | None],
    hooks: Mapping[str, Callable[..., Any]] | None = None,
) -> ContextManager[dict[str, str] | None]:
    if hooks:
        setattr(manager, _HOOKS_ATTR, dict(hooks))
    elif not hasattr(manager, _HOOKS_ATTR):
        setattr(manager, _HOOKS_ATTR, {})
    return manager


def _normalize_factory(factory: _FactoryCallable) -> FixtureFactory:
    def _as_context_manager() -> ContextManager[dict[str, str] | None]:
        result = factory()
        if isinstance(result, AbstractContextManager):
            return _attach_hooks(result)
        if isinstance(result, FixtureHandle):
            env_dict: dict[str, str] = result.env or {}
            hooks = result.hooks

            @contextmanager
            def _ctx() -> Iterator[dict[str, str] | None]:
                try:
                    yield env_dict
                finally:
                    if result.teardown:
                        result.teardown()

            return _attach_hooks(_ctx(), hooks)
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

            return _attach_hooks(_ctx())
        if result is None:

            @contextmanager
            def _ctx_none() -> Iterator[dict[str, str] | None]:
                yield {}

            return _attach_hooks(_ctx_none())
        if isinstance(result, dict):

            @contextmanager
            def _ctx_dict() -> Iterator[dict[str, str] | None]:
                yield result

            return _attach_hooks(_ctx_dict())
        raise TypeError(
            "fixture factory must return a context manager, FixtureHandle, dict,"
            " tuple[env, teardown], or None"
        )

    return _as_context_manager


def _wrap_factory(
    factory: FixtureFactory,
    *,
    name: str,
    force_teardown_log: bool,
) -> ContextManager[dict[str, str] | None]:
    @contextmanager
    def _logged_context() -> Iterator[dict[str, str] | None]:
        if force_teardown_log:
            setattr(factory, "tenzir_log_teardown", True)
        controller = FixtureController(name, factory)
        try:
            yield controller.start()
        finally:
            controller.stop()

    return _logged_context()


def _coerce_specs(items: Iterable[FixtureSpec | str]) -> tuple[FixtureSpec, ...]:
    """Normalize an iterable of fixture names or specs into a tuple of FixtureSpec."""
    result: list[FixtureSpec] = []
    for item in items:
        if isinstance(item, str):
            result.append(FixtureSpec(name=item))
        else:
            result.append(item)
    return tuple(result)


def _activate_into_stack(
    specs: tuple[FixtureSpec, ...],
    stack: ExitStack,
) -> dict[str, str]:
    combined: dict[str, str] = {}
    for spec in specs:
        factory = _FACTORIES.get(spec.name)
        if not factory:
            logger.debug("requested fixture '%s' has no registered factory", spec.name)
            continue
        force_teardown_log = bool(getattr(factory, "tenzir_log_teardown", False))
        env: dict[str, str] | None = stack.enter_context(
            _wrap_factory(factory, name=spec.name, force_teardown_log=force_teardown_log)
        )
        if env:
            combined.update(env)
    return combined


def _build_fixture_options_for_context(
    specs: tuple[FixtureSpec, ...],
    existing: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fixture-options mapping for context-aware fixture activation.

    Existing fixture options take precedence. Missing entries are populated from
    fixture specs, using a registered dataclass options type when available.
    """
    merged: dict[str, Any] = dict(existing)
    for spec in specs:
        if spec.name in merged:
            continue
        options_cls = _OPTIONS_CLASSES.get(spec.name)
        if options_cls is not None:
            try:
                merged[spec.name] = _instantiate_options(options_cls, spec.options)
            except TypeError as exc:
                raise ValueError(f"invalid options for fixture '{spec.name}': {exc}") from exc
            continue
        if spec.options:
            merged[spec.name] = spec.options
    return merged


def _push_fixture_options_context(
    specs: tuple[FixtureSpec, ...],
) -> Token[FixtureContext | None] | None:
    """Push a derived FixtureContext that includes options from fixture specs."""
    ctx = _CONTEXT.get()
    if ctx is None:
        return None
    existing_options = dict(ctx.fixture_options)
    merged_options = _build_fixture_options_for_context(specs, existing_options)
    if merged_options == existing_options:
        return None
    return _CONTEXT.set(dataclasses.replace(ctx, fixture_options=merged_options))


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


def register(
    name: str | None,
    factory: _FactoryCallable,
    *,
    replace: bool = False,
    options: type | None = None,
) -> None:
    resolved_name = _infer_name(factory, name)
    if resolved_name in _FACTORIES and not replace:
        raise ValueError(f"fixture '{resolved_name}' already registered")
    if options is not None:
        if not isinstance(options, type) or not dataclasses.is_dataclass(options):
            raise TypeError(
                f"'options' for fixture '{resolved_name}' must be a dataclass type, "
                f"got {type(options).__name__}"
            )
    _FACTORIES[resolved_name] = _normalize_factory(factory)
    if options is not None:
        _OPTIONS_CLASSES[resolved_name] = options


def fixture(
    func: _FactoryCallable | None = None,
    *,
    name: str | None = None,
    replace: bool = False,
    log_teardown: bool = False,
    options: type | None = None,
) -> Callable[[_FactoryCallable], _FactoryCallable] | _FactoryCallable:
    """Decorator registering a fixture factory.

    ``@fixture`` accepts generator functions, context managers, or callables that
    return any of the supported fixture factory types. When used on a generator
    function, the decorator implicitly wraps it with :func:`contextlib.contextmanager`
    so authors can ``yield`` environments directly.

    The optional ``options`` parameter accepts a frozen dataclass type. When
    provided, test authors can pass structured options in ``test.yaml`` and the
    fixture receives a typed instance via :func:`current_options`.
    """

    def _decorator(inner: _FactoryCallable) -> _FactoryCallable:
        resolved_name = _infer_name(inner, name)
        candidate: _FactoryCallable
        if inspect.isgeneratorfunction(inner):
            candidate = contextmanager(inner)
        else:
            candidate = inner

        register(resolved_name, candidate, replace=replace, options=options)

        if log_teardown:
            registered = _FACTORIES.get(resolved_name)
            if registered is not None:
                setattr(registered, "tenzir_log_teardown", True)

        return inner

    if func is not None:
        return _decorator(func)

    return _decorator


@contextmanager
def activate(names: Iterable[FixtureSpec | str]) -> Iterator[dict[str, str]]:
    normalized = _coerce_specs(names)
    scope = _SUITE_SCOPE.get()
    if scope is not None and scope.fixtures == normalized:
        scope.depth += 1
        try:
            yield scope.env
        finally:
            scope.depth -= 1
        return

    context_token = _push_fixture_options_context(normalized)
    stack = ExitStack()
    try:
        combined = _activate_into_stack(normalized, stack)
        yield combined
    finally:
        stack.close()
        if context_token is not None:
            _CONTEXT.reset(context_token)


@contextmanager
def suite_scope(names: Iterable[FixtureSpec | str]) -> Iterator[dict[str, str]]:
    normalized = _coerce_specs(names)
    existing = _SUITE_SCOPE.get()
    if existing is not None:
        raise RuntimeError("nested fixture suite scopes are not supported")

    context_token = _push_fixture_options_context(normalized)
    stack = ExitStack()
    combined = _activate_into_stack(normalized, stack)
    scope = _SuiteScope(fixtures=normalized, stack=stack, env=combined)
    token = _SUITE_SCOPE.set(scope)
    try:
        yield combined
    finally:
        _SUITE_SCOPE.reset(token)
        stack.close()
        if context_token is not None:
            _CONTEXT.reset(context_token)


# Import built-in fixtures so they self-register on package import.
from . import node  # noqa: F401,E402


__all__ = [
    "Executor",
    "FixtureContext",
    "FixtureHandle",
    "FixtureSelection",
    "FixtureSpec",
    "FixturesAccessor",
    "FixtureController",
    "FixtureUnavailable",
    "activate",
    "acquire_fixture",
    "current_options",
    "fixture",
    "fixtures",
    "fixtures_api",
    "get_options_class",
    "suite_scope",
    "has",
    "register",
    "require",
    "register_tmp_dir_cleanup",
    "unregister_tmp_dir_cleanup",
    "invoke_tmp_dir_cleanup",
]
