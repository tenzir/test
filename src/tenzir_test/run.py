#!/usr/bin/env python3


from __future__ import annotations

import atexit
import builtins
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import difflib
import enum
import fnmatch
import importlib.util
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeVar, cast, overload

import yaml

import tenzir_test.fixtures as fixtures_impl
from . import packages
from .config import Settings, discover_settings
from .runners import (
    ShellRunner,  # noqa: F401
    CustomPythonFixture,  # noqa: F401
    ExtRunner,  # noqa: F401
    Runner,  # noqa: F401
    TqlRunner,  # noqa: F401
    allowed_extensions,
    get_runner_for_test as runners_get_runner,
    iter_runners as runners_iter_runners,
    runner_names,
)


TestConfig = dict[str, object]

#: YAML value for the ``skip.on`` key that opts in to skipping a suite
#: when a fixture raises :class:`FixtureUnavailable`.
SKIP_ON_FIXTURE_UNAVAILABLE: typing.Final[str] = "fixture-unavailable"
SKIP_ON_CAPABILITY_UNAVAILABLE: typing.Final[str] = "capability-unavailable"


@dataclasses.dataclass(frozen=True, slots=True)
class SkipConfig:
    """Typed wrapper for the ``skip`` test configuration value.

    Replaces the raw ``str | dict | None`` representation with an explicit
    type so that consumers never need to perform ``isinstance`` checks
    themselves.

    Two forms are supported:

    * **Static skip** -- the test is unconditionally skipped with a reason
      string (``skip: "reason"`` in YAML).
    * **Conditional skip** -- the test is skipped when one of the configured
      suite conditions occurs (e.g. ``fixture-unavailable`` or
      ``capability-unavailable`` in directory-level ``test.yaml``).
    """

    reason: str | None = None
    on_fixture_unavailable: bool = False
    on_capability_unavailable: bool = False

    # -- query helpers -------------------------------------------------------

    @property
    def is_static(self) -> bool:
        """Return ``True`` when the test should be unconditionally skipped."""
        return (
            self.reason is not None
            and not self.on_fixture_unavailable
            and not self.on_capability_unavailable
        )

    @property
    def is_conditional(self) -> bool:
        """Return ``True`` when this is a conditional skip guard."""
        return self.on_fixture_unavailable or self.on_capability_unavailable

    def allows_condition(self, condition: str) -> bool:
        if condition == SKIP_ON_FIXTURE_UNAVAILABLE:
            return self.on_fixture_unavailable
        if condition == SKIP_ON_CAPABILITY_UNAVAILABLE:
            return self.on_capability_unavailable
        return False

    # -- factory -------------------------------------------------------------

    _VALID_DICT_KEYS: typing.ClassVar[frozenset[str]] = frozenset({"on", "reason"})
    _VALID_ON_VALUES: typing.ClassVar[frozenset[str]] = frozenset(
        {SKIP_ON_FIXTURE_UNAVAILABLE, SKIP_ON_CAPABILITY_UNAVAILABLE}
    )

    # PyYAML (YAML 1.1) parses bare ``on`` as the boolean ``True``.
    # Map such keys back to the intended string form before validation.
    _YAML_BOOL_KEY_MAP: typing.ClassVar[dict[object, str]] = {True: "on", False: "off"}

    @classmethod
    def _normalize_dict_keys(cls, raw: dict[object, object]) -> dict[str, object]:
        """Return a copy with YAML boolean keys replaced by their string equivalents."""
        out: dict[str, object] = {}
        for k, v in raw.items():
            if isinstance(k, bool):
                normalized = cls._YAML_BOOL_KEY_MAP.get(k)
                if normalized is None:
                    normalized = str(k).lower()
                out[normalized] = v
            else:
                out[str(k)] = v
        return out

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        origin: Literal["directory", "test"],
        location: "Path | str",
        line_number: int | None = None,
    ) -> "SkipConfig | None":
        """Parse a raw YAML value into a ``SkipConfig``.

        Returns ``None`` when *value* is ``None`` (skip not configured).
        Raises ``ValueError`` via ``_raise_config_error`` on invalid input.
        """
        if value is None:
            return None

        if isinstance(value, str):
            if not value.strip():
                _raise_config_error(
                    location,
                    "'skip' string value must be non-empty",
                    line_number,
                )
            return cls(reason=value)

        if isinstance(value, dict):
            if origin != "directory":
                _raise_config_error(
                    location,
                    f"'skip' mapping form (e.g. {{on: {SKIP_ON_FIXTURE_UNAVAILABLE}}}) "
                    "is only valid in directory-level test.yaml files, "
                    "not in test frontmatter",
                    line_number,
                )
            normalized = cls._normalize_dict_keys(value)
            unknown_keys = set(normalized.keys()) - cls._VALID_DICT_KEYS
            if unknown_keys:
                _raise_config_error(
                    location,
                    f"'skip' mapping contains unknown keys: "
                    f"{', '.join(sorted(unknown_keys))}; "
                    f"allowed keys are: {', '.join(sorted(cls._VALID_DICT_KEYS))}",
                    line_number,
                )
            on_value = normalized.get("on")
            on_values: list[str]
            if isinstance(on_value, str):
                on_values = [on_value]
            elif isinstance(on_value, list):
                if not on_value:
                    _raise_config_error(
                        location,
                        "'skip.on' list must contain at least one value",
                        line_number,
                    )
                if not all(isinstance(item, str) for item in on_value):
                    _raise_config_error(
                        location,
                        "'skip.on' list values must be strings",
                        line_number,
                    )
                on_values = cast(list[str], on_value)
            else:
                _raise_config_error(
                    location,
                    f"'skip.on' must be a string or list of strings, got '{type(on_value).__name__}'",
                    line_number,
                )
            normalized_on_values = [entry.strip() for entry in on_values]
            if not all(entry for entry in normalized_on_values):
                _raise_config_error(
                    location,
                    "'skip.on' values must be non-empty strings",
                    line_number,
                )
            invalid_values = [
                entry for entry in normalized_on_values if entry not in cls._VALID_ON_VALUES
            ]
            if invalid_values:
                _raise_config_error(
                    location,
                    f"'skip.on' must be one of "
                    f"{', '.join(repr(v) for v in sorted(cls._VALID_ON_VALUES))}, "
                    f"got '{invalid_values[0]}'",
                    line_number,
                )
            reason = normalized.get("reason")
            if reason is not None and (not isinstance(reason, str) or not reason.strip()):
                _raise_config_error(
                    location,
                    "'skip.reason' must be a non-empty string",
                    line_number,
                )
            return cls(
                reason=reason if isinstance(reason, str) else None,
                on_fixture_unavailable=SKIP_ON_FIXTURE_UNAVAILABLE in normalized_on_values,
                on_capability_unavailable=SKIP_ON_CAPABILITY_UNAVAILABLE in normalized_on_values,
            )

        _raise_config_error(
            location,
            f"'skip' must be a string or mapping {{on: ..., reason: ...}}, "
            f"got '{type(value).__name__}'",
            line_number,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RequiresConfig:
    """Typed wrapper for suite capability requirements."""

    operators: tuple[str, ...] = tuple()

    @property
    def has_requirements(self) -> bool:
        return bool(self.operators)

    def as_mapping(self) -> dict[str, tuple[str, ...]]:
        return {"operators": self.operators}

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        origin: Literal["directory", "test"],
        location: "Path | str",
        line_number: int | None = None,
    ) -> "RequiresConfig | None":
        if value is None:
            return None
        if origin != "directory":
            _raise_config_error(
                location,
                "'requires' can only be specified in directory-level test.yaml files",
                line_number,
            )
        if not isinstance(value, dict):
            _raise_config_error(
                location,
                f"'requires' must be a mapping {{operators: [...]}}, got '{type(value).__name__}'",
                line_number,
            )
        unknown_keys = set(value.keys()) - {"operators"}
        if unknown_keys:
            _raise_config_error(
                location,
                f"'requires' mapping contains unknown keys: "
                f"{', '.join(sorted(str(key) for key in unknown_keys))}; "
                "allowed keys are: operators",
                line_number,
            )
        raw_operators = value.get("operators")
        if raw_operators is None:
            return cls()
        if not isinstance(raw_operators, list):
            _raise_config_error(
                location,
                f"'requires.operators' must be a list of strings, got '{type(raw_operators).__name__}'",
                line_number,
            )
        operators: list[str] = []
        seen: set[str] = set()
        for entry in raw_operators:
            if not isinstance(entry, str):
                _raise_config_error(
                    location,
                    f"'requires.operators' entries must be strings, got '{type(entry).__name__}'",
                    line_number,
                )
            normalized = entry.strip()
            if not normalized:
                _raise_config_error(
                    location,
                    "'requires.operators' entries must be non-empty strings",
                    line_number,
                )
            if normalized in seen:
                continue
            seen.add(normalized)
            operators.append(normalized)
        return cls(operators=tuple(operators))


@dataclasses.dataclass(frozen=True, slots=True)
class RunSkippedSelector:
    """Fine-grained filter for selectively running skipped tests."""

    run_all: bool = False
    reason_patterns: tuple[str, ...] = tuple()

    @classmethod
    def from_cli(
        cls,
        *,
        run_all: bool = False,
        reason_patterns: Sequence[str] = (),
    ) -> "RunSkippedSelector":
        """Create a selector from CLI/library inputs."""

        normalized_patterns = tuple(
            pattern.strip() for pattern in reason_patterns if pattern and pattern.strip()
        )
        return cls(run_all=run_all, reason_patterns=normalized_patterns)

    @property
    def enabled(self) -> bool:
        return self.run_all or bool(self.reason_patterns)

    @property
    def has_reason_filters(self) -> bool:
        return bool(self.reason_patterns) and not self.run_all

    def _matches_reason_patterns(self, reason: str) -> bool:
        if not self.reason_patterns:
            return True
        for pattern in self.reason_patterns:
            if fnmatch.fnmatchcase(reason, _normalize_pattern(pattern)):
                return True
        return False

    def should_run_skipped(self, *, reason: str) -> bool:
        if not self.enabled:
            return False
        if self.run_all:
            return True
        return self._matches_reason_patterns(reason)

    def summary_flags(self) -> list[str]:
        flags: list[str] = []
        if self.run_all:
            flags.append("run-skipped")
            return flags
        if self.reason_patterns:
            flags.append(f"run-skipped-reason={len(self.reason_patterns)}")
        return flags


class SuiteExecutionMode(enum.Enum):
    """Supported execution modes for suite members."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteConfig:
    """Normalized suite configuration from directory-level `test.yaml`."""

    name: str
    mode: SuiteExecutionMode = SuiteExecutionMode.SEQUENTIAL


@dataclasses.dataclass(frozen=True, slots=True)
class TestQueueItem:
    runner: Runner
    path: Path


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteInfo:
    name: str
    directory: Path
    mode: SuiteExecutionMode = SuiteExecutionMode.SEQUENTIAL


@dataclasses.dataclass(slots=True)
class SuiteQueueItem:
    suite: SuiteInfo
    tests: list[TestQueueItem]
    fixtures: tuple[fixtures_impl.FixtureSpec, ...]


@dataclasses.dataclass(slots=True)
class SuiteCandidate:
    tests: list[TestQueueItem]
    fixtures: tuple[fixtures_impl.FixtureSpec, ...] | None = None
    parse_error: bool = False
    fixture_mismatch: bool = False
    mismatch_example: tuple[fixtures_impl.FixtureSpec, ...] | None = None
    mismatch_path: Path | None = None

    def record_fixtures(self, fixtures: tuple[fixtures_impl.FixtureSpec, ...]) -> None:
        if self.fixtures is None:
            self.fixtures = fixtures
            return
        if self.fixtures != fixtures:
            self.fixture_mismatch = True
            self.mismatch_example = fixtures
            if self.tests:
                self.mismatch_path = self.tests[-1].path

    def mark_parse_error(self) -> None:
        self.parse_error = True

    def is_valid(self) -> bool:
        return not self.parse_error and not self.fixture_mismatch


RunnerQueueItem = TestQueueItem | SuiteQueueItem


@dataclasses.dataclass(slots=True)
class ProjectSelection:
    """Describe which tests to execute for a given project root."""

    root: Path
    selectors: list[Path]
    run_all: bool
    kind: Literal["root", "satellite"]

    def should_run(self) -> bool:
        return self.run_all or bool(self.selectors)


@dataclasses.dataclass(slots=True)
class ExecutionPlan:
    """Aggregate the projects participating in a CLI invocation."""

    root: ProjectSelection
    satellites: list[ProjectSelection]

    def projects(self) -> Iterator[ProjectSelection]:
        yield self.root
        yield from self.satellites


T = TypeVar("T")


class ExecutionMode(enum.Enum):
    """Supported discovery modes."""

    PROJECT = "project"
    PACKAGE = "package"
    LIBRARY = "library"


class HarnessMode(enum.Enum):
    """Internal execution modes for the harness."""

    COMPARE = "compare"
    UPDATE = "update"
    PASSTHROUGH = "passthrough"


class ColorMode(enum.Enum):
    """Supported color output policies."""

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


def _is_library_root(path: Path) -> bool:
    """Return True when the directory looks like a library of packages."""

    if packages.is_package_dir(path):
        return False
    try:
        entries = list(packages.iter_package_dirs(path))
    except OSError:
        return False
    return bool(entries)


def detect_execution_mode(root: Path) -> tuple[ExecutionMode, Path | None]:
    """Return the execution mode and detected package root for `root`."""

    if packages.is_package_dir(root):
        return ExecutionMode.PACKAGE, root

    parent = root.parent if root.name == "tests" else None
    if parent is not None and packages.is_package_dir(parent):
        return ExecutionMode.PACKAGE, parent

    if _is_library_root(root):
        return ExecutionMode.LIBRARY, None

    return ExecutionMode.PROJECT, None


_settings: Settings | None = None
TENZIR_BINARY: tuple[str, ...] | None = None
TENZIR_NODE_BINARY: tuple[str, ...] | None = None
ROOT: Path = Path.cwd()
INPUTS_DIR: Path = ROOT / "inputs"
EXECUTION_MODE: ExecutionMode = ExecutionMode.PROJECT
_DETECTED_PACKAGE_ROOT: Path | None = None
HARNESS_MODE = HarnessMode.COMPARE
_COLOR_MODE = ColorMode.NEVER
COLORS_ENABLED = False
CHECKMARK = ""
CROSS = ""
INFO = ""
SKIP = ""
DEBUG_PREFIX = ""
BOLD = ""
CHECK_COLOR = ""
PASS_MAX_COLOR = ""
FAIL_COLOR = ""
SKIP_COLOR = ""
RESET_COLOR = ""
DETAIL_COLOR = ""
DIFF_ADD_COLOR = ""
PASS_SPECTRUM: list[str] = []
_COLORED_PASS_SPECTRUM = [
    "\033[38;5;52m",  # 0-9%   deep red
    "\033[38;5;88m",  # 10-19% red
    "\033[38;5;124m",  # 20-29% dark orange
    "\033[38;5;166m",  # 30-39% orange
    "\033[38;5;202m",  # 40-49% amber
    "\033[38;5;214m",  # 50-59% golden
    "\033[38;5;184m",  # 60-69% yellow-green
    "\033[38;5;148m",  # 70-79% spring green
    "\033[38;5;112m",  # 80-89% medium green
    "\033[38;5;28m",  # 90-99% deep forest green
    "\033[92m",  # 100% bright green
]
_INTERRUPTED_NOTICE = "└─▶ test interrupted by user"


def _colors_available() -> bool:
    if _COLOR_MODE is ColorMode.NEVER:
        return False
    if "NO_COLOR" in os.environ:
        return False
    if _COLOR_MODE is ColorMode.ALWAYS:
        return True
    return True


def _apply_color_palette() -> None:
    global COLORS_ENABLED
    global CHECKMARK
    global CROSS
    global INFO
    global SKIP
    global DEBUG_PREFIX
    global BOLD
    global CHECK_COLOR
    global PASS_MAX_COLOR
    global FAIL_COLOR
    global SKIP_COLOR
    global RESET_COLOR
    global DETAIL_COLOR
    global DIFF_ADD_COLOR
    global PASS_SPECTRUM
    global _INTERRUPTED_NOTICE

    COLORS_ENABLED = _colors_available()
    RESET_COLOR = "\033[0m" if COLORS_ENABLED else ""

    def _wrap(code: str, text: str) -> str:
        if not code:
            return text
        return f"{code}{text}{RESET_COLOR}"

    CHECK_COLOR = "\033[92;1m" if COLORS_ENABLED else ""
    PASS_MAX_COLOR = "\033[92m" if COLORS_ENABLED else ""
    FAIL_COLOR = "\033[31m" if COLORS_ENABLED else ""
    SKIP_COLOR = "\033[90;1m" if COLORS_ENABLED else ""
    DETAIL_COLOR = "\033[2;37m" if COLORS_ENABLED else ""
    DIFF_ADD_COLOR = "\033[32m" if COLORS_ENABLED else ""
    BOLD = "\033[1m" if COLORS_ENABLED else ""

    CHECKMARK = _wrap(CHECK_COLOR, "✔")
    CROSS = _wrap(FAIL_COLOR, "✘")
    INFO = _wrap("\033[94;1m" if COLORS_ENABLED else "", "i")
    SKIP = _wrap(SKIP_COLOR, "●")
    DEBUG_PREFIX = _wrap("\033[95m" if COLORS_ENABLED else "", "◆")
    PASS_SPECTRUM = (
        list(_COLORED_PASS_SPECTRUM) if COLORS_ENABLED else ["" for _ in _COLORED_PASS_SPECTRUM]
    )
    _INTERRUPTED_NOTICE = (
        f"└─▶ {_wrap('\033[33m' if COLORS_ENABLED else '', 'test interrupted by user')}"
    )


def refresh_color_palette() -> None:
    """Re-evaluate ANSI color availability based on environment variables."""

    _apply_color_palette()


def colors_enabled() -> bool:
    return COLORS_ENABLED


def get_color_mode() -> ColorMode:
    return _COLOR_MODE


def set_color_mode(mode: ColorMode) -> None:
    global _COLOR_MODE
    if not isinstance(mode, ColorMode):
        raise TypeError("mode must be an instance of ColorMode")
    if _COLOR_MODE is mode:
        return
    _COLOR_MODE = mode
    _apply_color_palette()


def colorize(text: str, color: str) -> str:
    """Wrap `text` with the given ANSI `color` code if colors are enabled."""

    if not color:
        return text
    return f"{color}{text}{RESET_COLOR}"


def format_failure_message(message: str) -> str:
    """Render a standardized failure line with optional ANSI coloring."""

    return f"└─▶ {colorize(message, FAIL_COLOR)}"


_apply_color_palette()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# Broader pattern covering all common ANSI escape sequences (CSI, OSC, etc.).
_ANSI_ESCAPE_FULL = re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|\[[^\x1b]*)")
# ASCII C0 control characters except tab (0x09) which is benign.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _sanitize_message(value: str) -> str:
    """Remove ANSI escape sequences and control characters from *value*.

    This is used to prevent untrusted exception messages from injecting
    terminal escape sequences or newlines into log/stdout output.
    """
    value = _ANSI_ESCAPE_FULL.sub("", value)
    value = _CONTROL_CHARS.sub("", value)
    return value


stdout_lock = threading.RLock()

_INTERRUPT_EVENT = threading.Event()
_INTERRUPT_ANNOUNCED = threading.Event()
_INTERRUPT_SIGNALS = {signal.SIGINT, signal.SIGTERM}


def interrupt_requested() -> bool:
    """Return whether a graceful shutdown was requested."""

    return _INTERRUPT_EVENT.is_set()


def _announce_interrupt() -> None:
    if _INTERRUPT_ANNOUNCED.is_set():
        return
    _INTERRUPT_ANNOUNCED.set()
    with stdout_lock:
        print(
            f"{INFO} received interrupt; finishing active tests (press Ctrl+C again to abort)",
        )


def _request_interrupt() -> None:
    first_interrupt = not _INTERRUPT_EVENT.is_set()
    _INTERRUPT_EVENT.set()
    if first_interrupt or not _INTERRUPT_ANNOUNCED.is_set():
        _announce_interrupt()


@contextlib.contextmanager
def _install_interrupt_handler() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(
        signum: int, frame: object | None
    ) -> None:  # pragma: no cover - signal path
        if interrupt_requested():
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        _request_interrupt()

    signal.signal(signal.SIGINT, _handle_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        _INTERRUPT_EVENT.clear()
        _INTERRUPT_ANNOUNCED.clear()


def _is_interrupt_exit(returncode: int) -> bool:
    if returncode < 0:
        return -returncode in _INTERRUPT_SIGNALS
    return returncode in {128 + sig for sig in _INTERRUPT_SIGNALS}


def _exception_is_interrupt(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)

        if isinstance(current, KeyboardInterrupt):
            return True

        returncode = getattr(current, "returncode", None)
        if isinstance(returncode, int) and _is_interrupt_exit(returncode):
            return True

        cause = current.__cause__
        if isinstance(cause, BaseException):
            pending.append(cause)
        context = current.__context__
        if isinstance(context, BaseException):
            pending.append(context)
    return False


_CURRENT_RETRY_CONTEXT = threading.local()
_CURRENT_SUITE_CONTEXT = threading.local()
_CURRENT_ASSERTION_CHECK_CONTEXT = threading.local()


@dataclasses.dataclass(slots=True)
class _AssertionCheckTally:
    total: int = 0
    failed: int = 0


@contextlib.contextmanager
def _push_suite_context(*, name: str, index: int, total: int) -> Iterator[None]:
    previous = getattr(_CURRENT_SUITE_CONTEXT, "value", None)
    _CURRENT_SUITE_CONTEXT.value = (name, index, total)
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_CURRENT_SUITE_CONTEXT, "value"):
                delattr(_CURRENT_SUITE_CONTEXT, "value")
        else:
            _CURRENT_SUITE_CONTEXT.value = previous


def _current_suite_progress() -> tuple[str, int, int] | None:
    value = getattr(_CURRENT_SUITE_CONTEXT, "value", None)
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and isinstance(value[2], int)
    ):
        return value
    return None


@contextlib.contextmanager
def _push_retry_context(*, attempt: int, max_attempts: int) -> Iterator[None]:
    previous = getattr(_CURRENT_RETRY_CONTEXT, "value", None)
    _CURRENT_RETRY_CONTEXT.value = (attempt, max_attempts)
    try:
        yield
    finally:
        setattr(_CURRENT_RETRY_CONTEXT, "last", (attempt, max_attempts))
        if previous is None:
            if hasattr(_CURRENT_RETRY_CONTEXT, "value"):
                delattr(_CURRENT_RETRY_CONTEXT, "value")
        else:
            _CURRENT_RETRY_CONTEXT.value = previous


def _current_retry_progress() -> tuple[int, int] | None:
    value = getattr(_CURRENT_RETRY_CONTEXT, "value", None)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
    ):
        return value
    fallback = getattr(_CURRENT_RETRY_CONTEXT, "last", None)
    if (
        isinstance(fallback, tuple)
        and len(fallback) == 2
        and isinstance(fallback[0], int)
        and isinstance(fallback[1], int)
    ):
        return fallback
    return None


@contextlib.contextmanager
def _push_assertion_check_context() -> Iterator[_AssertionCheckTally]:
    previous = getattr(_CURRENT_ASSERTION_CHECK_CONTEXT, "value", None)
    tally = _AssertionCheckTally()
    _CURRENT_ASSERTION_CHECK_CONTEXT.value = tally
    try:
        yield tally
    finally:
        if previous is None:
            if hasattr(_CURRENT_ASSERTION_CHECK_CONTEXT, "value"):
                delattr(_CURRENT_ASSERTION_CHECK_CONTEXT, "value")
        else:
            _CURRENT_ASSERTION_CHECK_CONTEXT.value = previous


def _record_assertion_check_result(*, failed: bool) -> None:
    tally = getattr(_CURRENT_ASSERTION_CHECK_CONTEXT, "value", None)
    if not isinstance(tally, _AssertionCheckTally):
        return
    tally.total += 1
    if failed:
        tally.failed += 1


def should_suppress_failure_output() -> bool:
    progress = getattr(_CURRENT_RETRY_CONTEXT, "value", None)
    if (
        isinstance(progress, tuple)
        and len(progress) == 2
        and isinstance(progress[0], int)
        and isinstance(progress[1], int)
    ):
        attempt, max_attempts = cast(tuple[int, int], progress)
        return attempt < max_attempts
    return False


def _format_attempt_suffix() -> str:
    progress = _current_retry_progress()
    if not progress:
        return ""
    attempt, max_attempts = progress
    if attempt <= 1 or max_attempts <= 1:
        return ""
    detail = f"attempts={attempt}/{max_attempts}"
    return f"  {colorize(detail, DETAIL_COLOR)}"


def _format_suite_suffix() -> str:
    progress = _current_suite_progress()
    if not progress:
        return ""
    name, index, total = progress
    if not name or total <= 0:
        return ""
    detail = f"suite={name} ({index}/{total})"
    return f"  {colorize(detail, DETAIL_COLOR)}"


TEST_TMP_ENV_VAR = "TENZIR_TMP_DIR"
_TMP_KEEP_ENV_VAR = "TENZIR_KEEP_TMP_DIRS"
_TMP_ROOT_NAME = ".tenzir-test"
_TMP_SUBDIR_NAME = "tmp"
_TMP_BASE_DIRS: set[Path] = set()
_ACTIVE_TMP_DIRS: set[Path] = set()
_TMP_DIR_LOCK = threading.Lock()
KEEP_TMP_DIRS = bool(os.environ.get(_TMP_KEEP_ENV_VAR))

SHOW_DIFF_OUTPUT = True
SHOW_DIFF_STAT = True
VERBOSE_OUTPUT = False
_BLOCK_INDENT = ""
_PLUS_SYMBOLS = {1: "□", 10: "▣", 100: "■"}
_MINUS_SYMBOLS = {1: "□", 10: "▣", 100: "■"}


def set_show_diff_output(enabled: bool) -> None:
    global SHOW_DIFF_OUTPUT
    SHOW_DIFF_OUTPUT = enabled


def should_show_diff_output() -> bool:
    return SHOW_DIFF_OUTPUT


def set_show_diff_stat(enabled: bool) -> None:
    global SHOW_DIFF_STAT
    SHOW_DIFF_STAT = enabled


def should_show_diff_stat() -> bool:
    return SHOW_DIFF_STAT


def set_verbose_output(enabled: bool) -> None:
    """Enable or disable verbose output for individual test results.

    When enabled, both success and skip messages are printed as tests complete.
    When disabled (default), only failures are printed.
    """
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = enabled


def is_verbose_output() -> bool:
    """Return whether verbose output is enabled."""
    return VERBOSE_OUTPUT


def set_harness_mode(mode: HarnessMode) -> None:
    """Set the global harness execution mode."""

    global HARNESS_MODE
    HARNESS_MODE = mode


def get_harness_mode() -> HarnessMode:
    """Return the current harness execution mode."""

    return HARNESS_MODE


def is_passthrough_enabled() -> bool:
    """Return whether passthrough output is enabled."""

    return HARNESS_MODE is HarnessMode.PASSTHROUGH


def is_update_mode() -> bool:
    """Return whether the harness updates reference artifacts."""

    return HARNESS_MODE is HarnessMode.UPDATE


def set_passthrough_enabled(enabled: bool) -> None:
    """Backward-compatible helper to toggle passthrough mode."""

    if enabled:
        set_harness_mode(HarnessMode.PASSTHROUGH)
    elif HARNESS_MODE is HarnessMode.PASSTHROUGH:
        set_harness_mode(HarnessMode.COMPARE)


@overload
def run_subprocess(
    args: Sequence[str],
    *,
    capture_output: bool,
    check: bool = False,
    text: Literal[False] = False,
    force_capture: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[bytes]: ...


@overload
def run_subprocess(
    args: Sequence[str],
    *,
    capture_output: bool,
    check: bool = False,
    text: Literal[True],
    force_capture: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]: ...


def run_subprocess(
    args: Sequence[str],
    *,
    capture_output: bool,
    check: bool = False,
    text: bool = False,
    force_capture: bool = False,
    stdin_data: bytes | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
    """Execute a subprocess honoring passthrough configuration.

    When passthrough is enabled the process inherits stdout/stderr so developers
    can observe output directly. Otherwise the helper captures both streams when
    `capture_output` is true, mirroring ``subprocess.run``'s behaviour.

    Runner authors should prefer this helper over direct ``subprocess`` calls so
    passthrough semantics remain consistent across implementations.

    Args:
        args: Command and arguments to execute.
        capture_output: Whether to capture stdout/stderr (ignored in passthrough mode).
        check: If True, raise CalledProcessError on non-zero exit.
        text: If True, decode stdout/stderr as text. Note: stdin_data must still
            be bytes even when text=True, as subprocess.run expects bytes for input.
        force_capture: Capture output even in passthrough mode.
        stdin_data: Bytes to send to the process's stdin. Use :func:`get_stdin_content`
            to read from a .stdin file. When None, stdin is not connected.
        **kwargs: Additional arguments passed to subprocess.run. The keys 'stdout',
            'stderr', 'capture_output', and 'input' are managed by this function
            and will raise TypeError if provided.

    Returns:
        A CompletedProcess instance with returncode, stdout, and stderr.

    Raises:
        TypeError: If stdout, stderr, capture_output, or input are passed in kwargs.
        subprocess.CalledProcessError: If check=True and process exits non-zero.
    """

    if any(key in kwargs for key in {"stdout", "stderr", "capture_output", "input"}):
        raise TypeError("run_subprocess manages stdout/stderr/input automatically")

    passthrough = is_passthrough_enabled()
    stream_output = passthrough and not force_capture
    stdout = subprocess.PIPE if capture_output and not stream_output else None
    stderr = subprocess.PIPE if capture_output and not stream_output else None

    if _CLI_LOGGER.isEnabledFor(logging.DEBUG):
        cmd_display = shlex.join(str(arg) for arg in args)
        cwd_value = kwargs.get("cwd")
        if cwd_value:
            cwd_segment = f" (cwd={cwd_value if isinstance(cwd_value, str) else str(cwd_value)})"
        else:
            cwd_segment = ""
        _CLI_LOGGER.debug("exec %s%s", cmd_display, cwd_segment)

    return subprocess.run(
        args,
        check=check,
        stdout=stdout,
        stderr=stderr,
        text=text,
        input=stdin_data,
        **kwargs,
    )


def _resolve_tmp_base() -> Path:
    preferred = ROOT / _TMP_ROOT_NAME / _TMP_SUBDIR_NAME
    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path(tempfile.gettempdir()) / _TMP_ROOT_NAME.strip(".") / _TMP_SUBDIR_NAME
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = preferred
    _TMP_BASE_DIRS.add(base)
    return base


def _tmp_prefix_for(test: Path) -> str:
    try:
        relative = test.relative_to(ROOT)
        base = relative.with_suffix("")
        candidate = "-".join(base.parts)
    except ValueError:
        candidate = test.stem
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "-", candidate)
    sanitized = sanitized.strip("-") or "test"
    return (sanitized[:32] if len(sanitized) > 32 else sanitized) or "test"


def _create_test_tmp_dir(test: Path) -> Path:
    prefix = f"{_tmp_prefix_for(test)}-"
    with _TMP_DIR_LOCK:
        base = _resolve_tmp_base()
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))
        _ACTIVE_TMP_DIRS.add(path)
    return path


def set_keep_tmp_dirs(enabled: bool) -> None:
    global KEEP_TMP_DIRS
    KEEP_TMP_DIRS = enabled


def cleanup_test_tmp_dir(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return
    tmp_path = Path(path)
    with _TMP_DIR_LOCK:
        _ACTIVE_TMP_DIRS.discard(tmp_path)
    try:
        fixtures_impl.invoke_tmp_dir_cleanup(tmp_path)
    except Exception:  # pragma: no cover - defensive logging
        pass
    if KEEP_TMP_DIRS:
        return
    with _TMP_DIR_LOCK:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        _cleanup_tmp_base_dirs()


def _cleanup_remaining_tmp_dirs() -> None:
    with _TMP_DIR_LOCK:
        remaining = list(_ACTIVE_TMP_DIRS)
    for tmp_path in remaining:
        cleanup_test_tmp_dir(tmp_path)
    with _TMP_DIR_LOCK:
        _cleanup_tmp_base_dirs()


def _cleanup_all_tmp_dirs() -> None:
    """Eagerly remove any temporary directories created by the harness."""

    if KEEP_TMP_DIRS:
        return
    with _TMP_DIR_LOCK:
        remaining = list(_ACTIVE_TMP_DIRS)
    for tmp_path in remaining:
        cleanup_test_tmp_dir(tmp_path)
    with _TMP_DIR_LOCK:
        _cleanup_tmp_base_dirs()


def _cleanup_tmp_base_dirs() -> None:
    if KEEP_TMP_DIRS:
        return
    for base in tuple(_TMP_BASE_DIRS):
        if not base.exists():
            _TMP_BASE_DIRS.discard(base)
            continue
        for candidate in _ACTIVE_TMP_DIRS:
            try:
                candidate.relative_to(base)
            except ValueError:
                continue
            break
        else:
            try:
                base.rmdir()
            except OSError:
                continue
            _TMP_BASE_DIRS.discard(base)
            parent = base.parent
            try:
                parent.rmdir()
            except OSError:
                pass


atexit.register(_cleanup_remaining_tmp_dirs)

_default_debug_logging = bool(os.environ.get("TENZIR_TEST_DEBUG"))
_debug_logging = _default_debug_logging

_runner_names: set[str] = set()
_allowed_extensions: set[str] = set()
_DEFAULT_RUNNER_BY_SUFFIX: dict[str, str] = {
    ".tql": "tenzir",
    ".py": "python",
    ".sh": "shell",
}

_CONFIG_FILE_NAME = "test.yaml"


class _CliDebugHandler(logging.Handler):
    """Stream debug messages through stdout using the CLI formatting."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        message = self.format(record)
        with stdout_lock:
            builtins.print(f"{DEBUG_PREFIX} {message}", flush=True)


_CLI_LOGGER = logging.getLogger("tenzir_test.cli")
if not _CLI_LOGGER.handlers:
    cli_handler = _CliDebugHandler()
    cli_handler.setLevel(logging.DEBUG)
    cli_handler.setFormatter(logging.Formatter("%(message)s"))
    _CLI_LOGGER.addHandler(cli_handler)
    _CLI_LOGGER.propagate = False
_CLI_LOGGER.setLevel(logging.DEBUG if _debug_logging else logging.WARNING)

_DIRECTORY_CONFIG_CACHE: dict[Path, "_DirectoryConfig"] = {}

_DISCOVERY_ENABLED = False
_CLI_PACKAGES: list[Path] = []


def _expand_package_dirs(path: Path) -> list[str]:
    """Normalize a package dir hint; if it contains packages, return those."""

    resolved = path.expanduser().resolve()
    if packages.is_package_dir(resolved):
        return [str(resolved)]
    expanded: list[str] = []
    try:
        for pkg in packages.iter_package_dirs(resolved):
            expanded.append(str(pkg.resolve()))
    except OSError as exc:
        _CLI_LOGGER.debug("failed to expand package dir %s: %s", path, exc)
        return [str(resolved)]
    return expanded or [str(resolved)]


def _set_discovery_logging(enabled: bool) -> None:
    global _DISCOVERY_ENABLED
    _DISCOVERY_ENABLED = enabled


def _set_cli_packages(package_paths: list[Path]) -> None:
    global _CLI_PACKAGES
    _CLI_PACKAGES = [path.resolve() for path in package_paths]


def _get_cli_packages() -> list[Path]:
    return list(_CLI_PACKAGES)


def _deduplicate_package_dirs(candidates: list[str]) -> list[str]:
    """Remove duplicate package directories while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser().resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(candidate)
    return result


def _print_discovery_message(message: str) -> None:
    if _CLI_LOGGER.isEnabledFor(logging.DEBUG):
        _CLI_LOGGER.debug(message)
    else:
        with stdout_lock:
            builtins.print(f"{DEBUG_PREFIX} {message}", flush=True)


class ProjectMarker(enum.Enum):
    """Sentinel indicators that describe a project root."""

    PACKAGE_MANIFEST = "package_manifest"
    TESTS_DIRECTORY = "tests_directory"
    TEST_CONFIG = "test_config"
    TEST_SUITE_DIRECTORY = "test_suite_directory"
    LIBRARY_ROOT = "library_root"


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectSignature:
    """Description of markers that identify a project root."""

    root: Path
    markers: frozenset[ProjectMarker]

    @property
    def kind(self) -> Literal["package", "project"]:
        return "package" if ProjectMarker.PACKAGE_MANIFEST in self.markers else "project"

    def has(self, marker: ProjectMarker) -> bool:
        return marker in self.markers


_PRIMARY_PROJECT_MARKERS = {
    ProjectMarker.PACKAGE_MANIFEST,
    ProjectMarker.TESTS_DIRECTORY,
    ProjectMarker.TEST_CONFIG,
    ProjectMarker.TEST_SUITE_DIRECTORY,
    ProjectMarker.LIBRARY_ROOT,
}


def _describe_project_root(path: Path) -> ProjectSignature | None:
    """Return a signature describing why a path qualifies as a project root."""

    try:
        resolved = path.resolve()
    except FileNotFoundError:
        return None

    if not resolved.exists() or not resolved.is_dir():
        return None

    markers: set[ProjectMarker] = set()

    if packages.is_package_dir(resolved):
        markers.add(ProjectMarker.PACKAGE_MANIFEST)

    tests_dir = resolved / "tests"
    if tests_dir.is_dir():
        markers.add(ProjectMarker.TESTS_DIRECTORY)

    if (resolved / "test.yaml").is_file():
        markers.add(ProjectMarker.TEST_CONFIG)

    if resolved.name == "tests" and resolved.is_dir():
        markers.add(ProjectMarker.TEST_SUITE_DIRECTORY)

    try:
        if _is_library_root(resolved):
            markers.add(ProjectMarker.LIBRARY_ROOT)
    except OSError:
        pass

    if not markers:
        return None

    if not markers & _PRIMARY_PROJECT_MARKERS:
        return None

    return ProjectSignature(root=resolved, markers=frozenset(markers))


def _discover_enclosed_projects(path: Path, *, base_root: Path) -> list[Path]:
    """Return project roots discovered directly underneath `path`."""

    try:
        resolved = path.resolve()
    except FileNotFoundError:
        return []

    if not resolved.exists() or not resolved.is_dir():
        return []

    candidates: list[Path] = []
    try:
        entries = sorted(resolved.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.is_dir():
            continue
        project_root = _find_project_root(entry, base_root=base_root)
        if project_root is None:
            continue
        resolved_root = project_root.resolve()
        if resolved_root == base_root:
            continue
        if resolved_root not in candidates:
            candidates.append(resolved_root)

    return candidates


def _set_project_root(path: Path) -> None:
    """Switch global project state to `path`."""

    global ROOT, INPUTS_DIR, EXECUTION_MODE, _DETECTED_PACKAGE_ROOT
    ROOT = path
    INPUTS_DIR = _resolve_inputs_dir(path)
    EXECUTION_MODE, _DETECTED_PACKAGE_ROOT = detect_execution_mode(path)
    _clear_directory_config_cache()


def _is_project_root(path: Path) -> bool:
    """Return True if the directory looks like a tenzir-test project root."""

    return _describe_project_root(path) is not None


def _resolve_cli_path(argument: Path, *, base_root: Path) -> Path:
    if argument.is_absolute():
        return argument.resolve()

    candidates = [Path.cwd() / argument, base_root / argument]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Neither candidate exists; prefer base-root resolution for error messages.
    return (base_root / argument).resolve()


def _find_project_root(path: Path, *, base_root: Path) -> Path | None:
    package_root = packages.find_package_root(path)
    if package_root is not None:
        return package_root.resolve()

    resolved = path.resolve()
    nested_test_dir = resolved / "test"
    if _is_project_root(nested_test_dir):
        resolved = nested_test_dir
    try:
        resolved.relative_to(base_root)
    except ValueError:
        pass
    else:
        if resolved != base_root and _is_project_root(resolved):
            try:
                rel = resolved.relative_to(base_root)
            except ValueError:
                return resolved
            if not rel.parts or rel.parts[0] != "tests":
                return resolved
        return base_root

    for candidate in [resolved, *resolved.parents]:
        if candidate == base_root:
            return base_root
        if candidate.name == "tests":
            parent = candidate.parent
            if parent == base_root:
                return base_root
            if _is_project_root(parent):
                return parent
        candidate_test_dir = candidate / "test"
        if (
            _is_project_root(candidate_test_dir)
            and candidate_test_dir != base_root
            and candidate_test_dir.is_relative_to(resolved)
        ):
            candidate = candidate_test_dir
        if _is_project_root(candidate):
            try:
                rel = candidate.relative_to(base_root)
            except ValueError:
                if base_root.is_relative_to(candidate):
                    continue
                return candidate
            if rel.parts and rel.parts[0] == "tests":
                continue
            return candidate
    return None


def _build_execution_plan(
    base_root: Path,
    raw_args: Sequence[Path],
    *,
    root_explicit: bool,
    all_projects: bool = False,
) -> ExecutionPlan:
    base_root_is_project = _is_project_root(base_root)
    root_selectors: list[Path] = []
    run_root_all = not raw_args

    satellite_order: list[Path] = []
    satellite_selectors: dict[Path, list[Path]] = {}
    satellite_run_all: dict[Path, bool] = {}

    for argument in raw_args:
        resolved = _resolve_cli_path(argument, base_root=base_root)
        project_root = _find_project_root(resolved, base_root=base_root)

        if project_root is None:
            if not resolved.exists():
                raise SystemExit(
                    f"error: path `{argument}` does not exist (resolved to {resolved})"
                )
            enclosed_projects = _discover_enclosed_projects(resolved, base_root=base_root)
            if enclosed_projects:
                for nested_root in enclosed_projects:
                    selectors = satellite_selectors.setdefault(nested_root, [])
                    if nested_root not in satellite_order:
                        satellite_order.append(nested_root)
                    satellite_run_all[nested_root] = True
                continue

            try:
                resolved.relative_to(base_root)
            except ValueError:
                if _DISCOVERY_ENABLED:
                    _print_discovery_message(
                        f"ignoring `{argument}` (resolved to {resolved}) - no tenzir-test project found"
                    )
                continue

            # Default to root project for existing paths inside the main project tree.
            project_root = base_root

        if project_root == base_root:
            if not base_root_is_project:
                enclosed_projects = _discover_enclosed_projects(resolved, base_root=base_root)
                if enclosed_projects:
                    for nested_root in enclosed_projects:
                        selectors = satellite_selectors.setdefault(nested_root, [])
                        if nested_root not in satellite_order:
                            satellite_order.append(nested_root)
                        satellite_run_all[nested_root] = True
                    continue
            if resolved == base_root:
                run_root_all = True
                continue
            if not resolved.exists():
                raise SystemExit(
                    f"error: test path `{argument}` does not exist (resolved to {resolved})"
                )
            root_selectors.append(resolved)
            continue

        project_root = project_root.resolve()
        selectors = satellite_selectors.setdefault(project_root, [])
        if project_root not in satellite_order:
            satellite_order.append(project_root)
        if resolved == project_root:
            satellite_run_all[project_root] = True
            continue
        if not resolved.exists():
            raise SystemExit(
                f"error: test path `{argument}` does not exist (resolved to {resolved})"
            )
        selectors.append(resolved)

    if all_projects:
        run_root_all = True
    elif root_explicit and not raw_args:
        run_root_all = True

    root_selection = ProjectSelection(
        root=base_root,
        selectors=[path.resolve() for path in root_selectors],
        run_all=run_root_all,
        kind="root",
    )

    satellites: list[ProjectSelection] = []
    for project_root in satellite_order:
        selectors = [path.resolve() for path in satellite_selectors.get(project_root, [])]
        run_all = satellite_run_all.get(project_root, False) or not selectors
        satellites.append(
            ProjectSelection(
                root=project_root,
                selectors=selectors,
                run_all=run_all,
                kind="satellite",
            )
        )

    return ExecutionPlan(root=root_selection, satellites=satellites)


def _format_relative_path(path: Path, base: Path) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError:
        try:
            relative_str = os.path.relpath(path, base)
        except ValueError:
            return path.as_posix()
        if relative_str == ".":
            return "."
        return relative_str.replace(os.sep, "/")
    if not relative.parts:
        return "."
    return relative.as_posix()


def _marker_for_selection(selection: ProjectSelection) -> str:
    is_package = packages.is_package_dir(selection.root)
    if selection.kind == "root":
        return "●" if is_package else "■"
    return "○" if is_package else "□"


def _print_execution_plan(plan: ExecutionPlan, *, display_base: Path) -> int:
    active: list[tuple[str, ProjectSelection]] = []
    if plan.root.should_run():
        active.append((_marker_for_selection(plan.root), plan.root))
    for satellite in plan.satellites:
        if satellite.should_run():
            active.append((_marker_for_selection(satellite), satellite))

    if not active:
        return 0

    if len(active) == 1:
        return 0

    print(f"{INFO} found {len(active)} projects")
    root_base = plan.root.root
    for marker, selection in active:
        if selection.kind == "satellite":
            name = _format_relative_path(selection.root, root_base)
        else:
            name = selection.root.name or selection.root.as_posix()
        print(f"{INFO}   {marker} {name}")
    return len(active)


_MISSING = object()


def get_default_jobs() -> int:
    """Return the default number of worker threads for the CLI."""

    return 4 * (os.cpu_count() or 16)


@dataclasses.dataclass(slots=True)
class _DirectoryConfig:
    values: TestConfig
    sources: dict[str, Path]


def _default_test_config() -> TestConfig:
    return {
        "error": False,
        "timeout": 30,
        "runner": None,
        "skip": None,
        "requires": None,
        "fixtures": tuple(),
        "assertions": {},
        "inputs": None,
        "retry": 1,
        "suite": None,
        "package_dirs": tuple(),
        "pre_compare": tuple(),
    }


def _suite_config_from_value(value: object) -> SuiteConfig | None:
    """Return a normalized SuiteConfig from raw config values."""

    if isinstance(value, SuiteConfig):
        return value
    if isinstance(value, str):
        name = value.strip()
        if not name:
            return None
        return SuiteConfig(name=name)
    if isinstance(value, Mapping):
        raw_name = value.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None
        raw_mode = value.get("mode", SuiteExecutionMode.SEQUENTIAL.value)
        if isinstance(raw_mode, SuiteExecutionMode):
            mode = raw_mode
        elif isinstance(raw_mode, str):
            try:
                mode = SuiteExecutionMode(raw_mode.strip().lower())
            except ValueError:
                return None
        else:
            return None
        return SuiteConfig(name=raw_name.strip(), mode=mode)
    return None


def _normalize_suite_value(
    value: object,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> SuiteConfig:
    """Normalize the `suite` config value from string or mapping form."""

    if isinstance(value, str):
        name = value.strip()
        if not name:
            _raise_config_error(
                location,
                "'suite' value must be a non-empty string",
                line_number,
            )
        return SuiteConfig(name=name)

    if isinstance(value, Mapping):
        unknown_keys = {str(key) for key in value.keys()} - {"name", "mode"}
        if unknown_keys:
            _raise_config_error(
                location,
                f"'suite' mapping contains unknown keys: {', '.join(sorted(unknown_keys))}; "
                "allowed keys are: name, mode",
                line_number,
            )
        raw_name = value.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            _raise_config_error(
                location,
                "'suite.name' value must be a non-empty string",
                line_number,
            )
        raw_mode = value.get("mode", SuiteExecutionMode.SEQUENTIAL.value)
        mode: SuiteExecutionMode
        if isinstance(raw_mode, SuiteExecutionMode):
            mode = raw_mode
        elif isinstance(raw_mode, str):
            try:
                mode = SuiteExecutionMode(raw_mode.strip().lower())
            except ValueError:
                valid_modes = ", ".join(mode.value for mode in SuiteExecutionMode)
                _raise_config_error(
                    location,
                    f"Invalid value for 'suite.mode', expected one of {valid_modes}, got '{raw_mode}'",
                    line_number,
                )
        else:
            _raise_config_error(
                location,
                f"Invalid value for 'suite.mode', expected string, got '{type(raw_mode).__name__}'",
                line_number,
            )
        return SuiteConfig(name=raw_name.strip(), mode=mode)

    _raise_config_error(
        location,
        f"Invalid value for 'suite', expected non-empty string or mapping, got '{type(value).__name__}'",
        line_number,
    )


def _canonical_config_key(key: str) -> str:
    if key == "fixture":
        return "fixtures"
    if key in {"package_dirs", "package-dirs"}:
        return "package_dirs"
    if key in {"pre_compare", "pre-compare"}:
        return "pre_compare"
    return key


ConfigOrigin = Literal["directory", "test"]


def _raise_config_error(
    location: Path | str, message: str, line_number: int | None = None
) -> typing.NoReturn:
    base = str(location)
    if line_number is not None:
        base = f"{base}:{line_number}"
    raise ValueError(f"Error in {base}: {message}")


def _normalize_fixtures_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> tuple[fixtures_impl.FixtureSpec, ...]:
    raw: typing.Any
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, list):
            raw = parsed
        else:
            raw = [value]
    else:
        _raise_config_error(
            location,
            f"Invalid value for 'fixtures', expected string or list, got '{value}'",
            line_number,
        )

    specs: list[fixtures_impl.FixtureSpec] = []
    for entry in raw:
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                _raise_config_error(
                    location,
                    "Fixture names must be non-empty strings",
                    line_number,
                )
            specs.append(fixtures_impl.FixtureSpec(name=name))
        elif isinstance(entry, dict):
            if len(entry) != 1:
                _raise_config_error(
                    location,
                    f"Fixture mapping must have exactly one key (the fixture name), got {list(entry.keys())}",
                    line_number,
                )
            name = next(iter(entry))
            if not isinstance(name, str) or not name.strip():
                _raise_config_error(
                    location,
                    f"Fixture name must be a non-empty string, got '{name}'",
                    line_number,
                )
            opts = entry[name]
            if not isinstance(opts, dict):
                _raise_config_error(
                    location,
                    f"Fixture options for '{name}' must be a mapping, got '{type(opts).__name__}'",
                    line_number,
                )
            specs.append(fixtures_impl.FixtureSpec(name=name.strip(), options=opts))
        else:
            _raise_config_error(
                location,
                f"Invalid fixture entry '{entry}', expected string or mapping",
                line_number,
            )
    return tuple(specs)


def _parse_cli_fixture_entry(raw: str) -> typing.Any:
    """Parse one ``--fixture`` entry into a value accepted by `_normalize_fixtures_value`."""

    text = raw.strip()
    if not text:
        return text

    shorthand_name, shorthand_sep, shorthand_remainder = text.partition(":")
    if (
        shorthand_sep
        and shorthand_name.strip()
        and shorthand_remainder.strip().startswith("{")
        and text.lstrip()[0] not in "{["
    ):
        try:
            options = yaml.safe_load(shorthand_remainder.strip())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid fixture options in '{raw}': {exc}") from exc
        if not isinstance(options, dict):
            raise ValueError(
                f"Fixture options in '{raw}' must be a mapping, got '{type(options).__name__}'"
            )
        return {shorthand_name.strip(): options}

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = text

    if isinstance(parsed, (list, dict)):
        return parsed

    if isinstance(parsed, str):
        return parsed

    return text


def _normalize_cli_fixture_specs(values: Sequence[str]) -> tuple[fixtures_impl.FixtureSpec, ...]:
    """Normalize repeated CLI ``--fixture`` values into fixture specs."""

    raw_entries: list[typing.Any] = []
    for raw in values:
        parsed = _parse_cli_fixture_entry(raw)
        if isinstance(parsed, list):
            raw_entries.extend(parsed)
        else:
            raw_entries.append(parsed)
    return _normalize_fixtures_value(raw_entries, location="--fixture")


def _fixture_names(specs: Iterable[fixtures_impl.FixtureSpec]) -> tuple[str, ...]:
    """Extract fixture names from a sequence of specs."""
    return tuple(spec.name for spec in specs)


def _build_fixture_options(
    specs: tuple[fixtures_impl.FixtureSpec, ...],
) -> dict[str, typing.Any]:
    """Convert fixture specs into the ``fixture_options`` mapping for FixtureContext.

    For every spec:
    - If the fixture has a registered options class: construct a typed instance
      with recursive nested dataclass instantiation.
    - If no options class but options were provided: store the raw dict.
    - If no options class and no options: omit from the dict.
    """
    result: dict[str, typing.Any] = {}
    for spec in specs:
        options_cls = fixtures_impl.get_options_class(spec.name)
        if options_cls is not None:
            try:
                result[spec.name] = fixtures_impl._instantiate_options(options_cls, spec.options)
            except TypeError as exc:
                raise ValueError(f"invalid options for fixture '{spec.name}': {exc}") from exc
        elif spec.options:
            result[spec.name] = spec.options
    return result


def _normalize_assertions_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> dict[str, dict[str, typing.Any]]:
    """Normalize the ``assertions`` config mapping.

    Expected schema:

    assertions:
      fixtures:
        <fixture-name>: <mapping>
    """
    parsed: typing.Any = value
    if isinstance(value, str):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None

    if parsed is None:
        return {}

    if not isinstance(parsed, dict):
        _raise_config_error(
            location,
            f"Invalid value for 'assertions', expected mapping, got '{type(parsed).__name__}'",
            line_number,
        )

    unknown_keys = set(parsed.keys()) - {"fixtures"}
    if unknown_keys:
        _raise_config_error(
            location,
            "Unknown keys in 'assertions': " + ", ".join(sorted(str(key) for key in unknown_keys)),
            line_number,
        )

    fixtures_raw = parsed.get("fixtures")
    if fixtures_raw is None:
        return {}
    if not isinstance(fixtures_raw, dict):
        _raise_config_error(
            location,
            f"Invalid value for 'assertions.fixtures', expected mapping, got '{type(fixtures_raw).__name__}'",
            line_number,
        )

    normalized: dict[str, dict[str, typing.Any]] = {}
    for fixture_name, fixture_assertions in fixtures_raw.items():
        if not isinstance(fixture_name, str) or not fixture_name.strip():
            _raise_config_error(
                location,
                f"Assertion fixture name must be a non-empty string, got '{fixture_name}'",
                line_number,
            )
        if fixture_assertions is None:
            normalized[fixture_name.strip()] = {}
            continue
        if not isinstance(fixture_assertions, dict):
            _raise_config_error(
                location,
                "Fixture assertions for "
                + f"'{fixture_name}' must be a mapping, got '{type(fixture_assertions).__name__}'",
                line_number,
            )
        normalized[fixture_name.strip()] = dict(fixture_assertions)
    return normalized


def _build_fixture_assertions(
    assertions: Mapping[str, Mapping[str, typing.Any]] | None,
) -> dict[str, typing.Any]:
    """Convert raw fixture assertions into typed payloads when registered."""
    if not assertions:
        return {}
    result: dict[str, typing.Any] = {}
    for fixture_name, payload in assertions.items():
        assertions_cls = fixtures_impl.get_assertions_class(fixture_name)
        if assertions_cls is not None:
            try:
                result[fixture_name] = fixtures_impl._instantiate_options(assertions_cls, payload)
            except TypeError as exc:
                raise ValueError(f"invalid assertions for fixture '{fixture_name}': {exc}") from exc
            continue
        result[fixture_name] = dict(payload)
    return result


def _extract_location_path(location: Path | str) -> Path:
    if isinstance(location, Path):
        return location
    location_str = str(location)
    if ":" in location_str:
        location_str = location_str.split(":", 1)[0]
    return Path(location_str)


def _normalize_inputs_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, os.PathLike) or isinstance(value, str):
        raw = os.fspath(value).strip()
        if not raw:
            _raise_config_error(
                location,
                "'inputs' value must be a non-empty string",
                line_number,
            )
        base_dir = _extract_location_path(location).parent
        path = Path(raw)
        if not path.is_absolute():
            path = base_dir / path
        try:
            normalized = path.resolve()
        except OSError:
            normalized = path
        return str(normalized)

    _raise_config_error(
        location,
        f"Invalid value for 'inputs', expected string, got '{value}'",
        line_number,
    )


def _normalize_package_dirs_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        _raise_config_error(
            location,
            f"Invalid value for 'package-dirs', expected list of strings, got '{value}'",
            line_number,
        )
    base_dir = _extract_location_path(location).parent
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, (str, os.PathLike)):
            _raise_config_error(
                location,
                f"Invalid package-dirs entry '{entry}', expected string",
                line_number,
            )
        raw = os.fspath(entry).strip()
        if not raw:
            _raise_config_error(
                location,
                "Invalid package-dirs entry: must be non-empty string",
                line_number,
            )
        path = Path(raw)
        if not path.is_absolute():
            path = base_dir / path
        try:
            path = path.resolve()
        except OSError:
            path = path
        normalized.append(str(path))
    return tuple(normalized)


def _normalize_pre_compare_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> tuple[str, ...]:
    entries: typing.Any
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, list):
            entries = parsed
        else:
            entries = [value]
    else:
        _raise_config_error(
            location,
            f"Invalid value for 'pre-compare', expected string or list, got '{value}'",
            line_number,
        )

    transforms: list[str] = []
    valid_names = set(_TRANSFORMS.keys())
    for entry in entries:
        if not isinstance(entry, str):
            _raise_config_error(
                location,
                f"Invalid pre-compare entry '{entry}', expected string",
                line_number,
            )
        name = entry.strip()
        if not name:
            _raise_config_error(
                location,
                "Pre-compare transform names must be non-empty strings",
                line_number,
            )
        if name not in valid_names:
            valid_list = ", ".join(sorted(valid_names))
            _raise_config_error(
                location,
                f"Unknown pre-compare transform '{name}', valid transforms: {valid_list}",
                line_number,
            )
        transforms.append(name)
    return tuple(transforms)


def _assign_config_option(
    config: TestConfig,
    key: str,
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
    origin: ConfigOrigin,
) -> None:
    canonical = _canonical_config_key(key)
    valid_keys: set[str] = {
        "error",
        "timeout",
        "runner",
        "skip",
        "requires",
        "fixtures",
        "assertions",
        "inputs",
        "retry",
        "package_dirs",
        "pre_compare",
    }
    if origin == "directory":
        valid_keys.add("suite")
    if canonical not in valid_keys:
        _raise_config_error(location, f"Unknown configuration key '{key}'", line_number)

    if canonical == "suite":
        if origin != "directory":
            _raise_config_error(
                location,
                "'suite' can only be specified in directory-level test.yaml files",
                line_number,
            )
        config[canonical] = _normalize_suite_value(
            value,
            location=location,
            line_number=line_number,
        )
        return

    if canonical == "skip":
        # Delegate all validation and parsing to SkipConfig.from_raw which
        # raises ValueError (via _raise_config_error) on invalid input.
        parsed_skip = SkipConfig.from_raw(
            value,
            origin=origin,
            location=location,
            line_number=line_number,
        )
        config[canonical] = parsed_skip
        return

    if canonical == "requires":
        parsed_requires = RequiresConfig.from_raw(
            value,
            origin=origin,
            location=location,
            line_number=line_number,
        )
        config[canonical] = parsed_requires
        return

    if canonical == "error":
        if isinstance(value, bool):
            config[canonical] = value
            return
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                config[canonical] = lowered == "true"
                return
        _raise_config_error(
            location,
            f"Invalid value for '{canonical}', expected 'true' or 'false', got '{value}'",
            line_number,
        )

    if canonical == "timeout":
        if isinstance(value, int):
            timeout_value = value
        elif isinstance(value, str) and value.strip().isdigit():
            timeout_value = int(value)
        else:
            _raise_config_error(
                location,
                f"Invalid value for 'timeout', expected integer, got '{value}'",
                line_number,
            )
        if timeout_value <= 0:
            _raise_config_error(
                location,
                f"Invalid value for 'timeout', expected positive integer, got '{value}'",
                line_number,
            )
        config[canonical] = timeout_value
        return

    if canonical == "fixtures":
        suite_value = _suite_config_from_value(config.get("suite"))
        if origin == "test" and suite_value is not None:
            _raise_config_error(
                location,
                "'fixtures' cannot be specified in test frontmatter within a suite; configure fixtures in test.yaml",
                line_number,
            )
        config[canonical] = _normalize_fixtures_value(
            value, location=location, line_number=line_number
        )
        return

    if canonical == "assertions":
        config[canonical] = _normalize_assertions_value(
            value, location=location, line_number=line_number
        )
        return

    if canonical == "inputs":
        config[canonical] = _normalize_inputs_value(
            value, location=location, line_number=line_number
        )
        return
    if canonical == "package_dirs":
        if origin != "directory":
            _raise_config_error(
                location,
                "'package-dirs' can only be specified in directory-level test.yaml files",
                line_number,
            )
        config[canonical] = _normalize_package_dirs_value(
            value, location=location, line_number=line_number
        )
        return
    if canonical == "retry":
        if origin == "test" and _suite_config_from_value(config.get("suite")) is not None:
            _raise_config_error(
                location,
                "'retry' cannot be overridden in test frontmatter within a suite",
                line_number,
            )
        if isinstance(value, int):
            retry_value = value
        elif isinstance(value, str) and value.strip().isdigit():
            retry_value = int(value)
        else:
            _raise_config_error(
                location,
                f"Invalid value for 'retry', expected integer, got '{value}'",
                line_number,
            )
        if retry_value <= 0:
            _raise_config_error(
                location,
                f"Invalid value for 'retry', expected positive integer, got '{value}'",
                line_number,
            )
        config[canonical] = retry_value
        return

    if canonical == "pre_compare":
        transforms = _normalize_pre_compare_value(value, location=location, line_number=line_number)
        config[canonical] = transforms
        return

    if canonical == "runner":
        if not isinstance(value, str):
            _raise_config_error(
                location,
                f"Invalid value for 'runner', expected string, got '{value}'",
                line_number,
            )
        runner_names = _runner_names or {runner.name for runner in runners_iter_runners()}
        if runner_names and value not in runner_names:
            _CLI_LOGGER.debug(
                "Runner '%s' is not registered; proceeding with explicit selection.",
                value,
            )
        config[canonical] = value
        return

    config[canonical] = value


def _format_config_value(value: object) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return repr(value)


def _log_directory_override(
    *,
    path: Path,
    key: str,
    previous: object,
    new: object,
    previous_source: Path,
) -> None:
    message = (
        f"{_relativize_path(path)} overrides '{key}'"
        f" from {_format_config_value(previous)} (defined in {_relativize_path(previous_source)})"
        f" to {_format_config_value(new)}"
    )
    _CLI_LOGGER.debug(message)


def _load_directory_config(directory: Path) -> _DirectoryConfig:
    resolved = directory.resolve()
    cached = _DIRECTORY_CONFIG_CACHE.get(resolved)
    if cached is not None:
        return cached

    try:
        resolved.relative_to(ROOT)
        inside_root = True
    except ValueError:
        inside_root = False

    sources: dict[str, Path]

    if inside_root and resolved != ROOT:
        parent_config = _load_directory_config(resolved.parent)
        values = dict(parent_config.values)
        sources = dict(parent_config.sources)
    else:
        values = _default_test_config()
        sources = {}

    config_path = resolved / _CONFIG_FILE_NAME
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Error in {config_path}: configuration must define a mapping")
        for raw_key, raw_value in data.items():
            key = _canonical_config_key(str(raw_key))
            previous_value = values.get(key, _MISSING)
            previous_source = sources.get(key)
            _assign_config_option(
                values,
                key,
                raw_value,
                location=config_path,
                origin="directory",
            )
            new_value = values.get(key)
            if (
                previous_source is not None
                and previous_value is not _MISSING
                and new_value != previous_value
            ):
                _log_directory_override(
                    path=config_path,
                    key=key,
                    previous=previous_value,
                    new=new_value,
                    previous_source=previous_source,
                )
            sources[key] = config_path

    directory_config = _DirectoryConfig(values=values, sources=sources)
    _DIRECTORY_CONFIG_CACHE[resolved] = directory_config
    return directory_config


def _get_directory_defaults(directory: Path) -> TestConfig:
    config = _load_directory_config(directory).values
    return dict(config)


def _resolve_suite_for_test(test: Path) -> SuiteInfo | None:
    directory_config = _load_directory_config(test.parent)
    suite_value = _suite_config_from_value(directory_config.values.get("suite"))
    if suite_value is None:
        return None
    suite_source = directory_config.sources.get("suite")
    if suite_source is None:
        return None
    suite_dir = suite_source.parent
    try:
        resolved_dir = suite_dir.resolve()
    except OSError:
        resolved_dir = suite_dir
    try:
        test.resolve().relative_to(resolved_dir)
    except (OSError, ValueError):
        # Only treat as a suite when the test lives under the suite directory.
        return None
    return SuiteInfo(name=suite_value.name, directory=resolved_dir, mode=suite_value.mode)


def _clear_directory_config_cache() -> None:
    _DIRECTORY_CONFIG_CACHE.clear()


def _iter_project_test_directories(root: Path) -> Iterator[Path]:
    """Yield directories that contain tests for the current project."""

    if EXECUTION_MODE is ExecutionMode.PACKAGE:
        if root.name == "tests" and root.is_dir():
            yield root
            return
        package_root = _DETECTED_PACKAGE_ROOT
        if package_root is None:
            return
        tests_dir = package_root / "tests"
        if tests_dir.is_dir():
            yield tests_dir
        return

    package_dirs = list(packages.iter_package_dirs(root))
    if package_dirs:
        for package_dir in package_dirs:
            tests_dir = package_dir / "tests"
            if tests_dir.is_dir():
                yield tests_dir
        if package_dirs:
            return

    default_tests = root / "tests"
    if default_tests.is_dir():
        yield default_tests
        return

    for dir_path in root.iterdir():
        if not dir_path.is_dir() or dir_path.name.startswith("."):
            continue
        if dir_path.name in {"fixtures", "runners"}:
            continue
        if _is_inputs_path(dir_path):
            continue
        yield dir_path


def _is_inputs_path(path: Path) -> bool:
    """Return True when the path lives under any inputs directory."""
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        parts = path.parts
    return "inputs" in parts


def _refresh_registry() -> None:
    global _runner_names, _allowed_extensions
    _runner_names = runner_names()
    _allowed_extensions = allowed_extensions()


def update_registry_metadata(names: list[str], extensions: list[str]) -> None:
    global _runner_names, _allowed_extensions
    _runner_names = set(names)
    _allowed_extensions = set(extensions)


def get_allowed_extensions() -> set[str]:
    return set(_allowed_extensions)


def default_runner_for_suffix(suffix: str) -> str | None:
    return _DEFAULT_RUNNER_BY_SUFFIX.get(suffix)


def _resolve_inputs_dir(root: Path) -> Path:
    direct = root / "inputs"
    if direct.exists():
        return direct
    tests_inputs = root / "tests" / "inputs"
    if tests_inputs.exists():
        return tests_inputs
    return direct


def _find_nearest_inputs_dir(test: Path, root: Path) -> Path | None:
    """Walk up from test directory to find nearest inputs/ directory.

    Returns the nearest inputs/ directory between the test and root,
    or None if no inputs/ directory exists in that range.
    """
    current = test.parent
    root_resolved = root.resolve()
    while True:
        candidate = current / "inputs"
        if candidate.is_dir():
            return candidate
        # Stop when we reach or pass the root
        if current.resolve() == root_resolved or current == current.parent:
            break
        current = current.parent
    return None


def _looks_like_project_root(path: Path) -> bool:
    """Return True when the path or one of its parents resembles a project root."""

    candidates = [path, *path.parents]
    for candidate in candidates:
        if packages.is_package_dir(candidate):
            return True
        if candidate.name == "tests" and candidate.is_dir():
            return True
        tests_dir = candidate / "tests"
        if tests_dir.is_dir():
            return True
    return False


def ensure_settings() -> Settings:
    """Return the active harness settings, discovering defaults on first use."""

    if _settings is None:
        apply_settings(discover_settings())
    return cast(Settings, _settings)


def apply_settings(settings: Settings) -> None:
    global TENZIR_BINARY, TENZIR_NODE_BINARY
    global _settings
    _settings = settings
    TENZIR_BINARY = settings.tenzir_binary
    TENZIR_NODE_BINARY = settings.tenzir_node_binary
    _set_project_root(settings.root)


def _import_module_from_path(module_name: str, path: Path, *, package: bool = False) -> ModuleType:
    if package:
        search_locations = [str(path.parent)]
    else:
        search_locations = None
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURE_LOAD_ROOTS: set[Path] = set()
_RUNNER_LOAD_ROOTS: set[Path] = set()


def _load_project_fixtures(root: Path, *, expose_namespace: bool) -> None:
    resolved_root = root.resolve()
    if resolved_root in _FIXTURE_LOAD_ROOTS:
        return

    fixtures_package = root / "fixtures"
    fixtures_file = root / "fixtures.py"

    try:
        alias_target = None
        if fixtures_package.is_dir():
            init_file = fixtures_package / "__init__.py"
            if init_file.exists():
                alias_target = _import_module_from_path(
                    "_tenzir_project_fixtures", init_file, package=True
                )
            else:
                for candidate in sorted(fixtures_package.glob("*.py")):
                    alias_target = _import_module_from_path(
                        f"_tenzir_project_fixture_{candidate.stem}", candidate
                    )
        elif fixtures_file.exists():
            alias_target = _import_module_from_path("_tenzir_project_fixtures", fixtures_file)
        if alias_target is not None and expose_namespace:
            if "fixtures" not in sys.modules:
                sys.modules["fixtures"] = alias_target
    except ValueError as exc:  # registration error (e.g., duplicate fixture)
        raise RuntimeError(f"failed to load fixtures from {resolved_root}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"failed to load fixtures from {resolved_root}: {exc}") from exc

    _FIXTURE_LOAD_ROOTS.add(resolved_root)


def _load_project_runners(root: Path, *, expose_namespace: bool) -> None:
    resolved_root = root.resolve()
    if resolved_root in _RUNNER_LOAD_ROOTS:
        return

    runners_package = root / "runners"

    alias_target = None

    try:
        if runners_package.is_dir():
            init_file = runners_package / "__init__.py"
            if init_file.exists():
                alias_target = _import_module_from_path(
                    "_tenzir_project_runners", init_file, package=True
                )
            else:
                for candidate in sorted(runners_package.glob("*.py")):
                    alias_target = _import_module_from_path(
                        f"_tenzir_project_runner_{candidate.stem}", candidate
                    )
        if alias_target is not None and expose_namespace:
            if "runners" not in sys.modules:
                sys.modules["runners"] = alias_target
    except ValueError as exc:
        raise RuntimeError(f"failed to load runners from {resolved_root}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"failed to load runners from {resolved_root}: {exc}") from exc

    _RUNNER_LOAD_ROOTS.add(resolved_root)
    _refresh_registry()


def get_test_env_and_config_args(
    test: Path,
    *,
    inputs: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    config_file = test.parent / "tenzir.yaml"
    node_config_file = test.parent / "tenzir-node.yaml"
    config_args = [f"--config={config_file}"] if config_file.exists() else []
    env = os.environ.copy()
    if inputs is None:
        # Try nearest inputs/ directory first, fall back to project-level
        nearest = _find_nearest_inputs_dir(test, ROOT)
        if nearest is not None:
            inputs_path = str(nearest.resolve())
        else:
            inputs_path = str(_resolve_inputs_dir(ROOT).resolve())
    else:
        candidate = Path(os.fspath(inputs))
        if not candidate.is_absolute():
            candidate = test.parent / candidate
        inputs_path = str(candidate.resolve())
    env["TENZIR_INPUTS"] = inputs_path
    # Check for inline input file (.input extension).
    # Note: .input and .stdin extensions are reserved; see RESERVED_EXTENSIONS
    # in tenzir_test.runners.ext_runner for the authoritative list.
    inline_input = test.with_suffix(".input")
    if inline_input.is_file():
        validated_input = _validate_path_within_root(inline_input, "input file")
        env["TENZIR_INPUT"] = str(validated_input)
    # Check for stdin file (.stdin extension)
    stdin_file = test.with_suffix(".stdin")
    if stdin_file.is_file():
        validated_stdin = _validate_path_within_root(stdin_file, "stdin file")
        env["TENZIR_STDIN"] = str(validated_stdin)
    if config_file.exists():
        env.setdefault("TENZIR_CONFIG", str(config_file))
    if node_config_file.exists():
        env["TENZIR_NODE_CONFIG"] = str(node_config_file)
    if TENZIR_BINARY:
        env["TENZIR_BINARY"] = shlex.join(TENZIR_BINARY)
    if TENZIR_NODE_BINARY:
        env["TENZIR_NODE_BINARY"] = shlex.join(TENZIR_NODE_BINARY)
    env["TENZIR_TEST_ROOT"] = str(ROOT)
    tmp_dir = _create_test_tmp_dir(test)
    env[TEST_TMP_ENV_VAR] = str(tmp_dir)
    return env, config_args


def _validate_path_within_root(path: Path, description: str) -> Path:
    """Validate that a resolved path is within the project ROOT.

    This provides defense-in-depth against symlink traversal attacks where a
    malicious .stdin or .input file could be a symlink pointing outside the
    project directory.

    Args:
        path: The path to validate (will be resolved to follow symlinks).
        description: Human-readable description for error messages (e.g., "stdin file").

    Returns:
        The resolved path if validation passes.

    Raises:
        RuntimeError: If the resolved path is outside the project root.
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        rel_path = _relativize_path(path)
        raise RuntimeError(f"{description} '{rel_path}' resolves outside project root") from None
    return resolved


def _apply_fixture_env(
    env: dict[str, str], fixtures: tuple[fixtures_impl.FixtureSpec, ...]
) -> None:
    if fixtures:
        env["TENZIR_TEST_FIXTURES"] = ",".join(spec.name for spec in fixtures)
    else:
        env.pop("TENZIR_TEST_FIXTURES", None)


def get_stdin_content(env: dict[str, str]) -> bytes | None:
    """Read stdin content from a file specified by TENZIR_STDIN environment variable.

    This function is intended for runner authors who need to provide stdin content
    to test processes. The TENZIR_STDIN variable is automatically set by the test
    harness when a ``.stdin`` file exists alongside the test file.

    Args:
        env: Environment dictionary, typically from :func:`get_test_env_and_config_args`.

    Returns:
        The file contents as bytes if TENZIR_STDIN is set and the file exists,
        otherwise None. Empty files return empty bytes (b"").

    Raises:
        RuntimeError: If TENZIR_STDIN is set but the file cannot be read
            (e.g., file not found, permission denied).

    Example:
        Runner authors can use this with :func:`run_subprocess`::

            env, config_args = get_test_env_and_config_args(test)
            stdin_content = get_stdin_content(env)
            completed = run_subprocess(
                ["my-command"],
                env=env,
                stdin_data=stdin_content,
                ...
            )
    """
    stdin_path = env.get("TENZIR_STDIN")
    if not stdin_path:
        return None
    try:
        return Path(stdin_path).read_bytes()
    except OSError as e:
        raise RuntimeError(f"Failed to read stdin file '{stdin_path}': {e.strerror}") from e


def set_debug_logging(enabled: bool) -> None:
    global _debug_logging
    _debug_logging = enabled
    _CLI_LOGGER.setLevel(logging.DEBUG if enabled else logging.WARNING)


def is_debug_logging_enabled() -> bool:
    return _debug_logging


def log_comparison(test: Path, ref_path: Path, *, mode: str) -> None:
    if not _debug_logging or should_suppress_failure_output():
        return
    rel_test = _relativize_path(test)
    rel_ref = _relativize_path(ref_path)
    _CLI_LOGGER.debug("%s %s -> %s", mode, rel_test, rel_ref)


def report_failure(test: Path, message: str) -> None:
    if should_suppress_failure_output():
        return
    with stdout_lock:
        fail(test)
        if message:
            print(message)


def report_interrupted_test(test: Path) -> None:
    """Emit a standardized message for user-triggered interrupts."""

    report_failure(test, _INTERRUPTED_NOTICE)


def parse_test_config(test_file: Path, coverage: bool = False) -> TestConfig:
    """Parse test configuration from frontmatter at the beginning of the file."""
    config = _default_test_config()
    explicit_test_runner = False

    defaults = _get_directory_defaults(test_file.parent)
    for key, value in defaults.items():
        config[key] = value

    is_tql = test_file.suffix == ".tql"
    comment_frontmatter_suffixes = {".py", ".sh"}
    is_comment_frontmatter = test_file.suffix in comment_frontmatter_suffixes

    def _error(message: str, line_number: int | None = None) -> None:
        location = f"{test_file}:{line_number}" if line_number is not None else f"{test_file}"
        raise ValueError(f"Error in {location}: {message}")

    with open(test_file, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    consumed_frontmatter = False
    if is_tql:
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx < len(lines) and lines[idx].strip() == "---":
            idx += 1
            yaml_lines: list[str] = []
            while idx < len(lines):
                line = lines[idx]
                if line.strip() == "---":
                    idx += 1
                    consumed_frontmatter = True
                    break
                yaml_lines.append(line)
                idx += 1
            if not consumed_frontmatter:
                _error("YAML frontmatter must be terminated with '---'")
            yaml_data = yaml.safe_load("".join(yaml_lines)) or {}
            if not isinstance(yaml_data, dict):
                _error("YAML frontmatter must define a mapping")
            for key, value in yaml_data.items():
                if _canonical_config_key(str(key)) == "runner":
                    explicit_test_runner = True
                _assign_config_option(
                    config,
                    str(key),
                    value,
                    location=test_file,
                    origin="test",
                )

    if not consumed_frontmatter and is_comment_frontmatter:
        line_number = 0
        for raw_line in lines:
            line_number += 1
            stripped = raw_line.strip()
            if line_number == 1 and stripped.startswith("#!"):
                # Skip shebangs so subsequent frontmatter comments still apply.
                continue
            if not stripped.startswith("#"):
                break
            content = stripped[1:].strip()
            parts = content.split(":", 1)
            if len(parts) != 2:
                if line_number == 1:
                    break
                _error("Invalid frontmatter, expected 'key: value'", line_number)
            key = parts[0].strip()
            value = parts[1].strip()
            if _canonical_config_key(key) == "runner":
                explicit_test_runner = True
            _assign_config_option(
                config,
                key,
                value,
                location=test_file,
                line_number=line_number,
                origin="test",
            )

    if coverage:
        timeout_value = cast(int, config["timeout"])
        config["timeout"] = timeout_value * 5

    suffix = test_file.suffix.lower()
    if suffix == ".sh" and not explicit_test_runner:
        shell_runner = _DEFAULT_RUNNER_BY_SUFFIX.get(".sh")
        if shell_runner is not None:
            config["runner"] = shell_runner

    runner_value = config.get("runner")
    if not isinstance(runner_value, str) or not runner_value:
        default_runner = _DEFAULT_RUNNER_BY_SUFFIX.get(suffix)
        if default_runner is None:
            matching_names = [
                runner.name
                for runner in runners_iter_runners()
                if getattr(runner, "_ext", None) == suffix.lstrip(".")
            ]
            if not matching_names:
                raise ValueError(
                    f"No runner registered for '{test_file}' (extension '{suffix or '<none>'}')"
                    " and no 'runner' specified in frontmatter"
                )
            default_runner = matching_names[0]
        config["runner"] = default_runner
    if config.get("suite") is None:
        config.pop("suite", None)
    return config


def print(*args: object, **kwargs: Any) -> None:
    # TODO: Properly solve the synchronization below.
    if "flush" not in kwargs:
        kwargs["flush"] = True
    builtins.print(*args, **kwargs)


@dataclasses.dataclass
class RunnerStats:
    total: int = 0
    failed: int = 0
    skipped: int = 0


@dataclasses.dataclass
class FixtureStats:
    total: int = 0
    failed: int = 0
    skipped: int = 0


def _merge_runner_stats(
    left: dict[str, RunnerStats], right: dict[str, RunnerStats]
) -> dict[str, RunnerStats]:
    merged: dict[str, RunnerStats] = {}
    for name in {**left, **right}:
        stats = RunnerStats()
        if (lhs := left.get(name)) is not None:
            stats.total += lhs.total
            stats.failed += lhs.failed
            stats.skipped += lhs.skipped
        if (rhs := right.get(name)) is not None:
            stats.total += rhs.total
            stats.failed += rhs.failed
            stats.skipped += rhs.skipped
        merged[name] = stats
    return merged


def _merge_fixture_stats(
    left: dict[str, FixtureStats], right: dict[str, FixtureStats]
) -> dict[str, FixtureStats]:
    merged: dict[str, FixtureStats] = {}
    for name in {**left, **right}:
        stats = FixtureStats()
        if (lhs := left.get(name)) is not None:
            stats.total += lhs.total
            stats.failed += lhs.failed
            stats.skipped += lhs.skipped
        if (rhs := right.get(name)) is not None:
            stats.total += rhs.total
            stats.failed += rhs.failed
            stats.skipped += rhs.skipped
        merged[name] = stats
    return merged


@dataclasses.dataclass
class Summary:
    failed: int = 0
    total: int = 0
    skipped: int = 0
    assertion_checks_total: int = 0
    assertion_checks_failed: int = 0
    failed_paths: list[Path] = dataclasses.field(default_factory=list)
    skipped_paths: list[Path] = dataclasses.field(default_factory=list)
    runner_stats: dict[str, RunnerStats] = dataclasses.field(default_factory=dict)
    fixture_stats: dict[str, FixtureStats] = dataclasses.field(default_factory=dict)

    def __add__(self, other: "Summary") -> "Summary":
        return Summary(
            failed=self.failed + other.failed,
            total=self.total + other.total,
            skipped=self.skipped + other.skipped,
            assertion_checks_total=self.assertion_checks_total + other.assertion_checks_total,
            assertion_checks_failed=self.assertion_checks_failed + other.assertion_checks_failed,
            failed_paths=[*self.failed_paths, *other.failed_paths],
            skipped_paths=[*self.skipped_paths, *other.skipped_paths],
            runner_stats=_merge_runner_stats(self.runner_stats, other.runner_stats),
            fixture_stats=_merge_fixture_stats(self.fixture_stats, other.fixture_stats),
        )

    def record_runner_outcome(self, runner_name: str, outcome: bool | str) -> None:
        stats = self.runner_stats.setdefault(runner_name, RunnerStats())
        stats.total += 1
        if outcome == "skipped":
            stats.skipped += 1
        elif not outcome:
            stats.failed += 1

    def record_fixture_outcome(self, fixtures: Iterable[str], outcome: bool | str) -> None:
        for fixture in fixtures:
            stats = self.fixture_stats.setdefault(fixture, FixtureStats())
            stats.total += 1
            if outcome == "skipped":
                stats.skipped += 1
            elif not outcome:
                stats.failed += 1

    def record_assertion_checks(self, *, total: int, failed: int) -> None:
        if total <= 0:
            return
        self.assertion_checks_total += total
        self.assertion_checks_failed += max(0, min(failed, total))


def _merge_summary_inplace(target: Summary, source: Summary) -> None:
    """Merge `source` counts into `target` while preserving list/dict fields."""

    merged = target + source
    target.failed = merged.failed
    target.total = merged.total
    target.skipped = merged.skipped
    target.assertion_checks_total = merged.assertion_checks_total
    target.assertion_checks_failed = merged.assertion_checks_failed
    target.failed_paths = merged.failed_paths
    target.skipped_paths = merged.skipped_paths
    target.runner_stats = merged.runner_stats
    target.fixture_stats = merged.fixture_stats


@dataclasses.dataclass(slots=True)
class ProjectResult:
    selection: ProjectSelection
    summary: Summary
    queue_size: int


@dataclasses.dataclass(slots=True)
class ExecutionResult:
    summary: Summary
    project_results: tuple[ProjectResult, ...]
    queue_size: int
    exit_code: int
    interrupted: bool


class HarnessError(RuntimeError):
    """Fatal harness error signalling invalid invocation or configuration."""

    def __init__(self, message: str, *, exit_code: int = 1, show_message: bool = True) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.show_message = show_message


def _format_percentage(count: int, total: int) -> str:
    return f"{_percentage_value(count, total)}%"


def _percentage_value(count: int, total: int) -> int:
    if total <= 0:
        return 0
    return int(round((count / total) * 100))


def _format_summary(summary: Summary) -> str:
    total = summary.total
    passed = max(0, total - summary.failed - summary.skipped)
    if total <= 0:
        return "Test summary: No tests were discovered."

    passed_segment = f"{CHECKMARK} Passed {passed}/{total} ({_format_percentage(passed, total)})"
    failed_segment = (
        f"{CROSS} Failed {summary.failed} ({_format_percentage(summary.failed, total)})"
    )
    skipped_segment = (
        f"{SKIP} Skipped {summary.skipped} ({_format_percentage(summary.skipped, total)})"
    )
    segments = [passed_segment, failed_segment, skipped_segment]
    if summary.assertion_checks_total > 0:
        assertion_passed = max(0, summary.assertion_checks_total - summary.assertion_checks_failed)
        assertion_segment = f"Assertions {assertion_passed}/{summary.assertion_checks_total}"
        segments.append(assertion_segment)

    return "Test summary: " + " • ".join(segments)


def _summarize_runner_plan(
    queue: Sequence[RunnerQueueItem],
    *,
    tenzir_version: str | None,
    runner_versions: Mapping[str, str] | None = None,
) -> str:
    breakdown = _runner_breakdown(
        queue,
        tenzir_version=tenzir_version,
        runner_versions=runner_versions,
    )
    if not breakdown:
        return "no runners"
    parts: list[str] = []
    for name, count, version in breakdown:
        base = name
        if version:
            base = f"{base} (v{version})"
        parts.append(f"{count}× {base}")
    return ", ".join(parts)


def _iter_queue_tests(queue: Sequence[RunnerQueueItem]) -> Iterator[TestQueueItem]:
    for item in queue:
        if isinstance(item, SuiteQueueItem):
            yield from item.tests
        else:
            yield item


def _suite_test_sort_key(directory: Path, path: Path) -> str:
    try:
        relative = path.relative_to(directory)
    except ValueError:
        return path.as_posix()
    return relative.as_posix()


def _queue_sort_key(item: RunnerQueueItem) -> str:
    if isinstance(item, SuiteQueueItem):
        if item.tests:
            return str(item.tests[0].path)
        return str(item.suite.directory)
    return str(item.path)


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _build_queue_from_paths(
    paths: Iterable[Path],
    *,
    coverage: bool,
) -> list[RunnerQueueItem]:
    suite_groups: dict[SuiteInfo, SuiteCandidate] = {}
    individuals: dict[Path, TestQueueItem] = {}

    for test_path in sorted({path.resolve() for path in paths}, key=lambda p: str(p)):
        try:
            runner = get_runner_for_test(test_path)
        except ValueError as error:
            raise HarnessError(f"error: {error}") from error

        suite_info = _resolve_suite_for_test(test_path)
        test_item = TestQueueItem(runner=runner, path=test_path)
        if suite_info is None:
            individuals[test_path] = test_item
            continue

        candidate = suite_groups.setdefault(suite_info, SuiteCandidate(tests=[]))
        candidate.tests.append(test_item)
        try:
            config = parse_test_config(test_path, coverage=coverage)
        except ValueError:
            candidate.mark_parse_error()
            continue
        fixtures = cast(tuple[fixtures_impl.FixtureSpec, ...], config.get("fixtures", tuple()))
        candidate.record_fixtures(fixtures)

    queue: list[RunnerQueueItem] = []
    for suite_info, candidate in suite_groups.items():
        if candidate.fixture_mismatch:
            example = candidate.mismatch_example or tuple()
            expected = candidate.fixtures or tuple()
            config_path = suite_info.directory / _CONFIG_FILE_NAME
            expected_list = ", ".join(str(s) for s in expected) or "<none>"
            example_list = ", ".join(str(s) for s in example) or "<none>"
            mismatch_path = candidate.mismatch_path or (
                candidate.tests[-1].path if candidate.tests else None
            )
            location_detail = (
                f" ({_relativize_path(mismatch_path)})" if mismatch_path is not None else ""
            )
            raise HarnessError(
                f"error: suite '{suite_info.name}' defined in {config_path} must use identical fixtures "
                f"across tests (expected: {expected_list}, found: {example_list}{location_detail})"
            )
        if not candidate.is_valid() or not candidate.tests:
            for test_item in candidate.tests:
                individuals[test_item.path] = test_item
            continue
        fixtures = candidate.fixtures or tuple()
        sorted_tests = sorted(
            candidate.tests,
            key=lambda item: _suite_test_sort_key(suite_info.directory, item.path),
        )
        queue.append(SuiteQueueItem(suite=suite_info, tests=sorted_tests, fixtures=fixtures))

    queue.extend(individuals.values())
    return queue


def _collect_runner_versions(
    queue: Sequence[RunnerQueueItem],
    *,
    tenzir_version: str | None,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    if tenzir_version:
        versions["tenzir"] = tenzir_version

    for item in _iter_queue_tests(queue):
        attr = getattr(item.runner, "version", None)
        if isinstance(attr, str) and attr:
            versions.setdefault(item.runner.name, attr)
    return versions


def _runner_breakdown(
    queue: Sequence[RunnerQueueItem],
    *,
    tenzir_version: str | None,
    runner_versions: Mapping[str, str] | None = None,
) -> list[tuple[str, int, str | None]]:
    counts: dict[str, int] = {}
    for item in _iter_queue_tests(queue):
        counts[item.runner.name] = counts.get(item.runner.name, 0) + 1

    breakdown: list[tuple[str, int, str | None]] = []
    for name in sorted(counts):
        version = (runner_versions or {}).get(name)
        if version is None and name == "tenzir":
            version = tenzir_version
        breakdown.append((name, counts[name], version))
    return breakdown


def _count_queue_tests(queue: Sequence[RunnerQueueItem]) -> int:
    total = 0
    for item in queue:
        if isinstance(item, SuiteQueueItem):
            total += len(item.tests)
        else:
            total += 1
    return total


def _print_aggregate_totals(project_count: int, summary: Summary) -> None:
    total = summary.total
    failed = summary.failed
    skipped = summary.skipped
    passed = total - failed - skipped
    executed = max(total - skipped, 0)
    project_noun = "project" if project_count == 1 else "projects"
    test_noun = "test" if total == 1 else "tests"
    if total <= 0:
        print(f"{INFO} ran 0 tests across {project_count} {project_noun}")
        return
    pass_rate = _percentage_value(passed, executed) if executed > 0 else 0
    fail_rate = _percentage_value(failed, executed) if executed > 0 else 0
    pass_index = min(pass_rate // 10, len(PASS_SPECTRUM) - 1)
    passed_percentage = f"{PASS_SPECTRUM[pass_index]}{pass_rate}%{RESET_COLOR}"
    if fail_rate > 0:
        failed_percentage = f"{FAIL_COLOR}{fail_rate}%{RESET_COLOR}"
    else:
        failed_percentage = f"{fail_rate}%"
    pass_segment = f"{passed} passed ({passed_percentage})"
    fail_segment = f"{failed} failed ({failed_percentage})"
    detail = f"{pass_segment} / {fail_segment}"
    if skipped:
        detail = f"{detail} • {skipped} skipped"
    print(f"{INFO} ran {total} {test_noun} across {project_count} {project_noun}: {detail}")


def _summarize_harness_configuration(
    *,
    jobs: int,
    update: bool,
    coverage: bool,
    debug: bool,
    show_summary: bool,
    runner_summary: bool,
    fixture_summary: bool,
    passthrough: bool,
    run_skipped_selector: RunSkippedSelector,
) -> tuple[int, str, str]:
    enabled_flags: list[str] = []
    toggles = (
        ("coverage", coverage),
        ("debug", debug),
        ("summary", show_summary),
        ("runner-summary", runner_summary),
        ("fixture-summary", fixture_summary),
        ("keep-tmp-dirs", KEEP_TMP_DIRS),
    )
    for name, flag in toggles:
        if flag:
            enabled_flags.append(name)
    enabled_flags.extend(run_skipped_selector.summary_flags())
    if update:
        verb = "updating"
    elif passthrough:
        verb = "showing"
    else:
        verb = "running"
    return jobs, ", ".join(enabled_flags), verb


def _relativize_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        try:
            relative = os.path.relpath(path, ROOT)
        except ValueError:
            return path
        return Path(relative)


def _get_test_fixtures(test: Path, *, coverage: bool) -> tuple[fixtures_impl.FixtureSpec, ...]:
    try:
        config = parse_test_config(test, coverage=coverage)
    except ValueError:
        return tuple()
    fixtures = config.get("fixtures", tuple())
    if isinstance(fixtures, tuple):
        return typing.cast(tuple[fixtures_impl.FixtureSpec, ...], fixtures)
    return tuple(typing.cast(Iterable[fixtures_impl.FixtureSpec], fixtures))


def _build_path_tree(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda p: p.parts):
        node = tree
        parts = path.parts
        if parts and parts[0] in {"..", "."}:
            parts = (path.as_posix(),)
        for part in parts:
            node = node.setdefault(part, {})
    return tree


def _render_tree(tree: dict[str, dict[str, Any]], prefix: str = "") -> Iterator[str]:
    items = sorted(tree.items())
    for index, (name, subtree) in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        yield f"{prefix}{connector}{name}"
        if subtree:
            extension = "    " if is_last else "│   "
            yield from _render_tree(subtree, prefix + extension)


def _strip_ansi(value: str) -> str:
    return ANSI_ESCAPE.sub("", value)


def _ljust_visible(value: str, width: int) -> str:
    visible = len(_strip_ansi(value))
    if visible >= width:
        return value
    return value + " " * (width - visible)


def _rjust_visible(value: str, width: int) -> str:
    visible = len(_strip_ansi(value))
    if visible >= width:
        return value
    return " " * (width - visible) + value


def _color_tree_glyphs(line: str, color: str) -> str:
    glyphs = {"│", "├", "└", "─"}
    parts = []
    for char in line:
        if char in glyphs and color:
            parts.append(f"{color}{char}{RESET_COLOR}")
        else:
            parts.append(char)
    return "".join(parts)


def _render_runner_box(summary: Summary) -> list[str]:
    if not summary.runner_stats:
        return []

    headers = ("Runner", "Passed", "Failed", "Skipped", "Total", "Share")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for name in sorted(summary.runner_stats):
        stats = summary.runner_stats[name]
        passed = max(0, stats.total - stats.failed - stats.skipped)
        rows.append(
            (
                name,
                str(passed),
                str(stats.failed),
                str(stats.skipped),
                str(stats.total),
                _format_percentage(stats.total, summary.total),
            )
        )

    label_width = max(len(headers[0]), *(len(_strip_ansi(row[0])) for row in rows))
    passed_width = max(len(headers[1]), *(len(row[1]) for row in rows))
    failed_width = max(len(headers[2]), *(len(row[2]) for row in rows))
    skipped_width = max(len(headers[3]), *(len(row[3]) for row in rows))
    total_width = max(len(headers[4]), *(len(row[4]) for row in rows))
    share_width = max(len(headers[5]), *(len(row[5]) for row in rows))

    def frame(char: str, width: int) -> str:
        return char * (width + 2)

    top = (
        f"┌{frame('─', label_width)}┬{frame('─', passed_width)}┬{frame('─', failed_width)}"
        f"┬{frame('─', skipped_width)}┬{frame('─', total_width)}┬{frame('─', share_width)}┐"
    )
    header = (
        f"│ {_ljust_visible(headers[0], label_width)} │ {headers[1].rjust(passed_width)} │ "
        f"{headers[2].rjust(failed_width)} │ {headers[3].rjust(skipped_width)} │ "
        f"{headers[4].rjust(total_width)} │ {headers[5].rjust(share_width)} │"
    )
    separator = (
        f"├{frame('─', label_width)}┼{frame('─', passed_width)}┼{frame('─', failed_width)}"
        f"┼{frame('─', skipped_width)}┼{frame('─', total_width)}┼{frame('─', share_width)}┤"
    )
    body = [
        f"│ {_ljust_visible(name, label_width)} │ {_rjust_visible(passed, passed_width)} │ "
        f"{_rjust_visible(failed, failed_width)} │ {_rjust_visible(skipped, skipped_width)} │ "
        f"{_rjust_visible(total, total_width)} │ {_rjust_visible(share, share_width)} │"
        for name, passed, failed, skipped, total, share in rows
    ]
    bottom = (
        f"└{frame('─', label_width)}┴{frame('─', passed_width)}┴{frame('─', failed_width)}"
        f"┴{frame('─', skipped_width)}┴{frame('─', total_width)}┴{frame('─', share_width)}┘"
    )
    return [top, header, separator, *body, bottom]


def _render_fixture_box(summary: Summary) -> list[str]:
    if not summary.fixture_stats:
        return []

    headers = ("Fixture", "Passed", "Failed", "Skipped", "Total", "Share")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for name in sorted(summary.fixture_stats):
        stats = summary.fixture_stats[name]
        passed = max(0, stats.total - stats.failed - stats.skipped)
        rows.append(
            (
                name,
                str(passed),
                str(stats.failed),
                str(stats.skipped),
                str(stats.total),
                _format_percentage(stats.total, summary.total),
            )
        )

    label_width = max(len(headers[0]), *(len(_strip_ansi(row[0])) for row in rows))
    passed_width = max(len(headers[1]), *(len(row[1]) for row in rows))
    failed_width = max(len(headers[2]), *(len(row[2]) for row in rows))
    skipped_width = max(len(headers[3]), *(len(row[3]) for row in rows))
    total_width = max(len(headers[4]), *(len(row[4]) for row in rows))
    share_width = max(len(headers[5]), *(len(row[5]) for row in rows))

    def frame(char: str, width: int) -> str:
        return char * (width + 2)

    top = (
        f"┌{frame('─', label_width)}┬{frame('─', passed_width)}┬{frame('─', failed_width)}"
        f"┬{frame('─', skipped_width)}┬{frame('─', total_width)}┬{frame('─', share_width)}┐"
    )
    header = (
        f"│ {_ljust_visible(headers[0], label_width)} │ {headers[1].rjust(passed_width)} │ "
        f"{headers[2].rjust(failed_width)} │ {headers[3].rjust(skipped_width)} │ "
        f"{headers[4].rjust(total_width)} │ {headers[5].rjust(share_width)} │"
    )
    separator = (
        f"├{frame('─', label_width)}┼{frame('─', passed_width)}┼{frame('─', failed_width)}"
        f"┼{frame('─', skipped_width)}┼{frame('─', total_width)}┼{frame('─', share_width)}┤"
    )
    body = [
        f"│ {_ljust_visible(name, label_width)} │ {_rjust_visible(passed, passed_width)} │ "
        f"{_rjust_visible(failed, failed_width)} │ {_rjust_visible(skipped, skipped_width)} │ "
        f"{_rjust_visible(total, total_width)} │ {_rjust_visible(share, share_width)} │"
        for name, passed, failed, skipped, total, share in rows
    ]
    bottom = (
        f"└{frame('─', label_width)}┴{frame('─', passed_width)}┴{frame('─', failed_width)}"
        f"┴{frame('─', skipped_width)}┴{frame('─', total_width)}┴{frame('─', share_width)}┘"
    )
    return [top, header, separator, *body, bottom]


def _render_summary_box(summary: Summary) -> list[str]:
    total = summary.total
    if total <= 0:
        return [
            "┌──────────────────────────┐",
            "│ No tests were discovered │",
            "└──────────────────────────┘",
        ]

    passed = max(0, total - summary.failed - summary.skipped)
    executed = max(total - summary.skipped, 0)
    pass_share = _percentage_value(passed, executed) if executed > 0 else 0
    fail_share = _percentage_value(summary.failed, executed) if executed > 0 else 0
    skip_share = _percentage_value(summary.skipped, total)

    executed_rows = [
        (f"{CHECKMARK} Passed", str(passed), f"{pass_share}%"),
        (f"{CROSS} Failed", str(summary.failed), f"{fail_share}%"),
    ]
    skipped_row = (f"{SKIP} Skipped", str(summary.skipped), f"{skip_share}%")
    total_row = ("∑ Total", str(total), "")

    all_rows = [*executed_rows, skipped_row, total_row]
    headers = ("Outcome", "Count")
    label_width = max(len(headers[0]), *(len(_strip_ansi(row[0])) for row in all_rows))
    count_lengths = [len(_strip_ansi(row[1])) for row in all_rows]
    count_width = max(count_lengths, default=0)
    share_lengths = [len(row[2]) for row in all_rows if row[2]]
    percent_width = max(share_lengths, default=0)
    spacing = 2 if percent_width else 0
    column_width = count_width + spacing + percent_width

    def frame(char: str, width: int) -> str:
        return char * (width + 2)

    top = f"┌{frame('─', label_width)}┬{frame('─', column_width)}┐"
    header = f"│ {_ljust_visible(headers[0], label_width)} │ {_ljust_visible(headers[1], column_width)} │"
    separator = f"├{frame('─', label_width)}┼{frame('─', column_width)}┤"
    executed_lines = [
        f"│ {_ljust_visible(label, label_width)} │ "
        f"{_rjust_visible(count, count_width)}"
        f"{' ' * spacing if percent_width else ''}"
        f"{_rjust_visible(percent, percent_width) if percent_width else ''} │"
        for label, count, percent in executed_rows
    ]
    group_separator = f"├{frame('─', label_width)}┼{frame('─', column_width)}┤"
    skipped_line = (
        f"│ {_ljust_visible(skipped_row[0], label_width)} │ "
        f"{_rjust_visible(skipped_row[1], count_width)}"
        f"{' ' * spacing if percent_width else ''}"
        f"{_rjust_visible(skipped_row[2], percent_width) if percent_width else ''} │"
    )
    summary_separator = f"├{frame('─', label_width)}┼{frame('─', column_width)}┤"
    total_line = (
        f"│ {_ljust_visible(total_row[0], label_width)} │ "
        f"{_ljust_visible(total_row[1], column_width)} │"
    )
    bottom = f"└{frame('─', label_width)}┴{frame('─', column_width)}┘"
    return [
        top,
        header,
        separator,
        *executed_lines,
        group_separator,
        skipped_line,
        summary_separator,
        total_line,
        bottom,
    ]


def _render_assertion_checks_box(summary: Summary) -> list[str]:
    if summary.assertion_checks_total <= 0:
        return []

    passed = max(0, summary.assertion_checks_total - summary.assertion_checks_failed)
    headers = ("Assertion Checks", "Passed", "Failed", "Total")
    row = (
        headers[0],
        str(passed),
        str(summary.assertion_checks_failed),
        str(summary.assertion_checks_total),
    )
    label_width = max(len(headers[0]), len(row[0]))
    passed_width = max(len(headers[1]), len(row[1]))
    failed_width = max(len(headers[2]), len(row[2]))
    total_width = max(len(headers[3]), len(row[3]))

    def frame(char: str, width: int) -> str:
        return char * (width + 2)

    top = (
        f"┌{frame('─', label_width)}┬{frame('─', passed_width)}"
        f"┬{frame('─', failed_width)}┬{frame('─', total_width)}┐"
    )
    header = (
        f"│ {_ljust_visible(headers[0], label_width)} │ {headers[1].rjust(passed_width)} │ "
        f"{headers[2].rjust(failed_width)} │ {headers[3].rjust(total_width)} │"
    )
    separator = (
        f"├{frame('─', label_width)}┼{frame('─', passed_width)}"
        f"┼{frame('─', failed_width)}┼{frame('─', total_width)}┤"
    )
    body = (
        f"│ {_ljust_visible(row[0], label_width)} │ {_rjust_visible(row[1], passed_width)} │ "
        f"{_rjust_visible(row[2], failed_width)} │ {_rjust_visible(row[3], total_width)} │"
    )
    bottom = (
        f"└{frame('─', label_width)}┴{frame('─', passed_width)}"
        f"┴{frame('─', failed_width)}┴{frame('─', total_width)}┘"
    )
    return [top, header, separator, body, bottom]


def _print_ascii_summary(summary: Summary, *, include_runner: bool, include_fixture: bool) -> None:
    runner_lines = _render_runner_box(summary) if include_runner else []
    fixture_lines = _render_fixture_box(summary) if include_fixture else []
    outcome_lines = _render_summary_box(summary)
    assertion_lines = _render_assertion_checks_box(summary)
    if not runner_lines and not fixture_lines and not outcome_lines and not assertion_lines:
        return
    print()
    segments: list[list[str]] = []
    if runner_lines:
        segments.append(runner_lines)
    if fixture_lines:
        segments.append(fixture_lines)
    if outcome_lines:
        segments.append(outcome_lines)
    if assertion_lines:
        segments.append(assertion_lines)
    for index, block in enumerate(segments):
        for line in block:
            print(line)
        if index < len(segments) - 1:
            print()


def _print_compact_summary(summary: Summary) -> None:
    total = summary.total
    passed = max(0, total - summary.failed - summary.skipped)
    executed = max(total - summary.skipped, 0)
    if total > 0:
        pass_rate = _percentage_value(passed, executed) if executed > 0 else 0
        fail_rate = _percentage_value(summary.failed, executed) if executed > 0 else 0
        pass_index = min(pass_rate // 10, len(PASS_SPECTRUM) - 1)
        pass_percentage = f"{PASS_SPECTRUM[pass_index]}{pass_rate}%{RESET_COLOR}"
        if fail_rate > 0:
            fail_percentage = f"{FAIL_COLOR}{fail_rate}%{RESET_COLOR}"
        else:
            fail_percentage = f"{fail_rate}%"
        pass_segment = f"{passed} passed ({pass_percentage})"
        fail_segment = f"{summary.failed} failed ({fail_percentage})"
        detail = f"{pass_segment} / {fail_segment}"
        if summary.skipped:
            detail = f"{detail} • {summary.skipped} skipped"
        if summary.assertion_checks_total > 0:
            assertion_passed = max(
                0, summary.assertion_checks_total - summary.assertion_checks_failed
            )
            detail = f"{detail} • assertions {assertion_passed}/{summary.assertion_checks_total}"
        noun = "test" if total == 1 else "tests"
        print(f"{INFO} ran {total} {noun}: {detail}")
    else:
        print(f"{INFO} ran 0 tests")


def _print_detailed_summary(summary: Summary) -> None:
    if not summary.failed_paths and not summary.skipped_paths:
        return

    print()
    if summary.skipped_paths:
        print(f"{SKIP} Skipped tests:")
        for line in _render_tree(_build_path_tree(summary.skipped_paths)):
            print(f"  {line}")
        if summary.failed_paths:
            print()
    if summary.failed_paths:
        print(f"{CROSS} Failed tests:")
        for line in _render_tree(_build_path_tree(summary.failed_paths)):
            print(f"  {_color_tree_glyphs(line, FAIL_COLOR)}")


def get_version() -> str:
    if not TENZIR_BINARY:
        raise FileNotFoundError("TENZIR_BINARY is not configured")
    return (
        subprocess.check_output(
            [
                *TENZIR_BINARY,
                "--bare-mode",
                "--console-verbosity=warning",
                "version | select version | write_lines",
            ]
        )
        .decode()
        .strip()
    )


def success(test: Path) -> None:
    if not is_verbose_output():
        return
    with stdout_lock:
        rel_test = _relativize_path(test)
        suite_suffix = _format_suite_suffix()
        attempt_suffix = _format_attempt_suffix()
        print(f"{CHECKMARK} {rel_test}{suite_suffix}{attempt_suffix}")


# Failures always print regardless of verbose mode because failures are critical
# information that users need to see immediately. In contrast, success() and
# handle_skip() respect the verbose setting since passed/skipped tests are
# expected outcomes that can be summarized at the end.
def fail(test: Path) -> None:
    with stdout_lock:
        rel_test = _relativize_path(test)
        attempt_suffix = _format_attempt_suffix()
        suite_suffix = _format_suite_suffix()
        print(f"{CROSS} {rel_test}{suite_suffix}{attempt_suffix}")


def last_and(items: Iterable[T]) -> Iterator[tuple[bool, T]]:
    iterator = iter(items)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for item in iterator:
        yield (False, previous)
        previous = item


def _format_unary_symbols(count: int, symbols: dict[int, str]) -> str:
    if count <= 0:
        return ""
    hundreds, remainder = divmod(count, 100)
    tens, ones = divmod(remainder, 10)
    parts: list[str] = []
    if hundreds:
        parts.append(symbols[100] * hundreds)
    if tens:
        parts.append(symbols[10] * tens)
    if ones:
        parts.append(symbols[1] * ones)
    return "".join(parts)


def _format_diff_counter(added: int, removed: int) -> str:
    plus_segment = _format_unary_symbols(added, _PLUS_SYMBOLS)
    minus_segment = _format_unary_symbols(removed, _MINUS_SYMBOLS)
    colored_plus = colorize(plus_segment, DIFF_ADD_COLOR) if plus_segment else ""
    colored_minus = colorize(minus_segment, FAIL_COLOR) if minus_segment else ""
    return f"{colored_plus}{colored_minus}"


def _format_stat_header(path: os.PathLike[str] | str, added: int, removed: int) -> str:
    path_str = os.fspath(path)
    counter = _format_diff_counter(added, removed)
    plus_count = colorize(f"{added}(+)", DIFF_ADD_COLOR)
    minus_count = colorize(f"{removed}(-)", FAIL_COLOR)
    if counter:
        counter_segment = f" {counter}"
    else:
        counter_segment = ""
    return f"{_BLOCK_INDENT}┌ {path_str} {plus_count}/{minus_count}{counter_segment}"


def _format_lines_changed(total: int) -> str:
    line = "line" if total == 1 else "lines"
    return f"{_BLOCK_INDENT}└ {total} {line} changed"


def _transform_sort(output: bytes) -> bytes:
    """Sort output lines lexicographically.

    Uses surrogateescape to preserve undecodable bytes as surrogate escapes,
    allowing the transform to handle binary data gracefully.
    """
    if not output:
        return output
    has_trailing_newline = output.endswith(b"\n")
    text = output.decode("utf-8", errors="surrogateescape")
    lines = text.splitlines(keepends=False)
    sorted_lines = sorted(lines)
    result = "\n".join(sorted_lines)
    if has_trailing_newline:
        result += "\n"
    return result.encode("utf-8", errors="surrogateescape")


# Transforms are intentionally simple and hardcoded rather than using the plugin
# architecture (like runners and fixtures). Rationale:
# - Transforms are core comparison utilities, not user-extensible features
# - Currently only one transform exists; extensibility can be added if needed
# - Pre-compare transforms are rarely customized per-project compared to runners
# - If custom transforms become necessary, this can be refactored to use a
#   registry pattern similar to runners/__init__.py
_TRANSFORMS: dict[str, typing.Callable[[bytes], bytes]] = {
    "sort": _transform_sort,
}


def apply_pre_compare(output: bytes, transforms: tuple[str, ...]) -> bytes:
    """Apply pre-compare transforms in order."""
    for name in transforms:
        output = _TRANSFORMS[name](output)
    return output


def print_diff(expected: bytes, actual: bytes, path: Path) -> None:
    if should_suppress_failure_output():
        return
    diff = list(
        difflib.diff_bytes(
            difflib.unified_diff,
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            n=2,
        )
    )
    added = sum(
        1
        for index, line in enumerate(diff)
        if index >= 2 and line.startswith(b"+") and not line.startswith(b"+++")
    )
    removed = sum(
        1
        for index, line in enumerate(diff)
        if index >= 2 and line.startswith(b"-") and not line.startswith(b"---")
    )
    show_stat = should_show_diff_stat()
    show_diff = should_show_diff_output()
    diff_lines: list[str] = []
    if should_show_diff_output():
        skip = 2
        for raw_line in diff:
            if skip > 0:
                skip -= 1
                continue
            text = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if raw_line.startswith(b"+") and not raw_line.startswith(b"+++"):
                text = colorize(text, DIFF_ADD_COLOR)
            elif raw_line.startswith(b"-") and not raw_line.startswith(b"---"):
                text = colorize(text, FAIL_COLOR)
            diff_lines.append(text)
    rel_path = _relativize_path(path)
    rel_path_str = os.fspath(rel_path)
    lines: list[str] = []
    total_changed = added + removed
    if not show_stat and not show_diff:
        return
    header = (
        _format_stat_header(rel_path, added, removed)
        if show_stat
        else f"{_BLOCK_INDENT}┌ {rel_path_str}"
    )
    lines.append(header)
    if show_diff and diff_lines:
        for diff_line in diff_lines:
            lines.append(f"{_BLOCK_INDENT}│ {diff_line}")
    if show_stat or (show_diff and total_changed > 0):
        lines.append(_format_lines_changed(total_changed))
    with stdout_lock:
        for output_line in lines:
            print(output_line)


def check_group_is_empty(pgid: int) -> None:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    raise ValueError("leftover child processes!")


def run_simple_test(
    test: Path,
    *,
    update: bool,
    args: Sequence[str] = (),
    output_ext: str,
    coverage: bool = False,
) -> bool:
    try:
        # Parse test configuration
        test_config = parse_test_config(test, coverage=coverage)
    except ValueError as e:
        report_failure(test, format_failure_message(str(e)))
        return False

    inputs_override = typing.cast(str | None, test_config.get("inputs"))
    env, config_args = get_test_env_and_config_args(test, inputs=inputs_override)
    fixture_specs = cast(
        tuple[fixtures_impl.FixtureSpec, ...], test_config.get("fixtures", tuple())
    )
    fixture_assertions_raw = cast(
        Mapping[str, Mapping[str, Any]] | None,
        test_config.get("assertions"),
    )
    timeout = cast(int, test_config["timeout"])
    expect_error = bool(test_config.get("error", False))
    passthrough_mode = is_passthrough_enabled()

    config_package_dirs = cast(tuple[str, ...], test_config.get("package_dirs", tuple()))
    additional_package_dirs: list[str] = []
    for entry in config_package_dirs:
        additional_package_dirs.extend(_expand_package_dirs(Path(entry)))

    package_root = packages.find_package_root(test)
    package_args: list[str] = []
    package_dir_candidates: list[str] = []
    if package_root is not None:
        env["TENZIR_PACKAGE_ROOT"] = str(package_root)
        package_tests_root = package_root / "tests"
        if inputs_override is None:
            # Try nearest inputs/ directory, fall back to package-level
            nearest = _find_nearest_inputs_dir(test, package_root)
            if nearest is not None:
                env["TENZIR_INPUTS"] = str(nearest.resolve())
            else:
                env["TENZIR_INPUTS"] = str(package_tests_root / "inputs")
        package_dir_candidates.append(str(package_root))
    package_dir_candidates.extend(additional_package_dirs)
    for cli_path in _get_cli_packages():
        package_dir_candidates.extend(_expand_package_dirs(cli_path))
    if package_dir_candidates:
        merged_dirs = _deduplicate_package_dirs(package_dir_candidates)
        env["TENZIR_PACKAGE_DIRS"] = ",".join(merged_dirs)
        package_args.append(f"--package-dirs={','.join(merged_dirs)}")

    fixture_options = _build_fixture_options(fixture_specs)
    fixture_assertions = _build_fixture_assertions(fixture_assertions_raw)
    context_token = fixtures_impl.push_context(
        fixtures_impl.FixtureContext(
            test=test,
            config=cast(dict[str, Any], test_config),
            coverage=coverage,
            env=env,
            config_args=tuple(config_args),
            tenzir_binary=TENZIR_BINARY,
            tenzir_node_binary=TENZIR_NODE_BINARY,
            fixture_options=fixture_options,
            fixture_assertions=fixture_assertions,
        )
    )
    try:
        with fixtures_impl.activate(fixture_specs) as fixture_env:
            env.update(fixture_env)
            _apply_fixture_env(env, fixture_specs)

            # Set up environment for code coverage if enabled
            if coverage:
                coverage_dir = os.environ.get(
                    "CMAKE_COVERAGE_OUTPUT_DIRECTORY", os.path.join(os.getcwd(), "coverage")
                )
                source_dir = os.environ.get("COVERAGE_SOURCE_DIR", os.getcwd())
                os.makedirs(coverage_dir, exist_ok=True)
                test_name = test.stem
                profile_path = os.path.join(coverage_dir, f"{test_name}-%p.profraw")
                env["LLVM_PROFILE_FILE"] = profile_path
                env["COVERAGE_SOURCE_DIR"] = source_dir

            node_args: list[str] = []
            fixture_names = _fixture_names(fixture_specs)
            node_requested = "node" in fixture_names
            if node_requested:
                endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                if not endpoint:
                    raise RuntimeError("node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT")
                node_args.append(f"--endpoint={endpoint}")

            if not TENZIR_BINARY:
                raise RuntimeError("TENZIR_BINARY must be configured before running tests")
            cmd: list[str] = [
                *TENZIR_BINARY,
                "--bare-mode",
                "--console-verbosity=warning",
                "--multi",
                *config_args,
                *node_args,
                *package_args,
                *args,
                "-f",
                str(test),
            ]
            stdin_content = get_stdin_content(env)
            completed = run_subprocess(
                cmd,
                timeout=timeout,
                env=env,
                capture_output=not passthrough_mode,
                cwd=str(ROOT),
                stdin_data=stdin_content,
            )

            good = completed.returncode == 0
            output = b""
            stderr_output = b""
            if not passthrough_mode:
                root_bytes = str(ROOT).encode() + b"/"
                captured_stdout = completed.stdout or b""
                output = captured_stdout.replace(root_bytes, b"")
                captured_stderr = completed.stderr or b""
                stderr_output = captured_stderr.replace(root_bytes, b"")

            if expect_error == good:
                interrupted = _is_interrupt_exit(completed.returncode) or interrupt_requested()
                if should_suppress_failure_output() and not interrupted:
                    return False
                if interrupted:
                    _request_interrupt()
                summary_line = (
                    _INTERRUPTED_NOTICE
                    if interrupted
                    else format_failure_message(f"got unexpected exit code {completed.returncode}")
                )
                if passthrough_mode:
                    report_failure(test, summary_line)
                else:
                    with stdout_lock:
                        fail(test)
                        if not interrupted:
                            line_prefix = "│ ".encode()
                            for line in output.splitlines():
                                sys.stdout.buffer.write(line_prefix + line + b"\n")
                            if completed.returncode != 0 and stderr_output:
                                sys.stdout.write("├─▶ stderr\n")
                                detail_prefix = DETAIL_COLOR.encode()
                                reset_bytes = RESET_COLOR.encode()
                                for line in stderr_output.splitlines():
                                    sys.stdout.buffer.write(
                                        line_prefix + detail_prefix + line + reset_bytes + b"\n"
                                    )
                        if summary_line:
                            sys.stdout.write(summary_line + "\n")
                return False
            if passthrough_mode:
                if not fixtures_impl.is_suite_scope_active(fixture_specs):
                    try:
                        _run_fixture_assertions_for_test(
                            test=test,
                            fixture_specs=fixture_specs,
                            fixture_assertions=fixture_assertions,
                        )
                    except Exception as exc:
                        report_failure(test, _fixture_assertion_failure_message(exc))
                        return False
                success(test)
                return True
            if not good:
                output_ext = "txt"
            ref_path = test.with_suffix(f".{output_ext}")
            if update:
                with ref_path.open("wb") as f:
                    f.write(output)
            else:
                if not ref_path.exists():
                    report_failure(
                        test, format_failure_message(f'Failed to find ref file: "{ref_path}"')
                    )
                    return False
                log_comparison(test, ref_path, mode="comparing")
                expected = ref_path.read_bytes()
                pre_compare = cast(tuple[str, ...], test_config.get("pre_compare", tuple()))
                expected_transformed = apply_pre_compare(expected, pre_compare)
                output_transformed = apply_pre_compare(output, pre_compare)
                if expected_transformed != output_transformed:
                    if interrupt_requested():
                        report_interrupted_test(test)
                    else:
                        report_failure(test, "")
                        print_diff(expected_transformed, output_transformed, ref_path)
                    return False
            if not fixtures_impl.is_suite_scope_active(fixture_specs):
                try:
                    _run_fixture_assertions_for_test(
                        test=test,
                        fixture_specs=fixture_specs,
                        fixture_assertions=fixture_assertions,
                    )
                except Exception as exc:
                    report_failure(test, _fixture_assertion_failure_message(exc))
                    return False
            success(test)
            return True
    except subprocess.TimeoutExpired:
        report_failure(
            test,
            format_failure_message(f"subprocess hit {timeout}s timeout"),
        )
        return False
    except subprocess.CalledProcessError as e:
        report_failure(test, format_failure_message(f"subprocess error: {e}"))
        return False
    except Exception as e:
        report_failure(test, format_failure_message(f"unexpected exception: {e}"))
        return False
    finally:
        fixtures_impl.pop_context(context_token)
        cleanup_test_tmp_dir(env.get(TEST_TMP_ENV_VAR))


def _compose_skip_reason(
    static_reason: str | None,
    exception_reason: str | None,
    *,
    fallback: str = "fixture unavailable",
) -> str:
    """Build a human-readable skip reason from up to two sources.

    Handles four cases:
    - Both a static config reason and an exception message are present: join
      them with ``": "``.
    - Only a static config reason is present: use it as-is.
    - Only an exception message is present: use it as-is.
    - Neither is present: return the *fallback* string.
    The *exception_reason* is sanitized to strip ANSI escape sequences
    and control characters before use.
    """
    if exception_reason:
        exception_reason = _sanitize_message(exception_reason)
    has_static = bool(static_reason)
    has_exc = bool(exception_reason)
    if has_static and has_exc:
        return f"{static_reason}: {exception_reason}"
    if has_static:
        return static_reason  # type: ignore[return-value]
    if has_exc:
        return exception_reason  # type: ignore[return-value]
    return fallback


def _run_fixture_assertions_for_test(
    *,
    test: Path,
    fixture_specs: tuple[fixtures_impl.FixtureSpec, ...],
    fixture_assertions: Mapping[str, Any],
) -> None:
    """Invoke fixture `assert_test` hooks for the given test."""
    if not fixture_specs:
        return
    try:
        fixtures_impl.invoke_active_hook(
            "assert_test",
            fixture_names=_fixture_names(fixture_specs),
            test=test,
            assertions_by_fixture=fixture_assertions,
            on_result=lambda success: _record_assertion_check_result(failed=not success),
        )
    except Exception as exc:
        raise RuntimeError(f"fixture assertion failed: {_sanitize_message(str(exc))}") from exc


def _fixture_assertion_failure_message(exc: BaseException) -> str:
    detail = _sanitize_message(str(exc)).strip()
    if not detail:
        detail = exc.__class__.__name__
    prefix = "fixture assertion failed:"
    if detail.startswith(prefix):
        return format_failure_message(detail)
    return format_failure_message(f"{prefix} {detail}")


@contextlib.contextmanager
def _push_fixture_test_context(
    *,
    test: Path,
    config: TestConfig,
    fixture_assertions: Mapping[str, Any],
) -> Iterator[None]:
    """Temporarily bind per-test fixture assertions into the active context."""
    current = fixtures_impl.current_context()
    if current is None:
        yield
        return
    token = fixtures_impl.push_context(
        dataclasses.replace(
            current,
            test=test,
            config=cast(dict[str, Any], config),
            fixture_assertions=fixture_assertions,
        )
    )
    try:
        yield
    finally:
        fixtures_impl.pop_context(token)


def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
    if is_verbose_output():
        rel_path = _relativize_path(test)
        suite_suffix = _format_suite_suffix()
        print(f"{SKIP} skipped {rel_path}{suite_suffix}: {reason}")
    ref_path = test.with_suffix(f".{output_ext}")
    if update:
        with ref_path.open("wb") as f:
            f.write(b"")
    else:
        if ref_path.exists():
            expected = ref_path.read_bytes()
            if expected != b"":
                report_failure(
                    test,
                    format_failure_message(
                        f'Reference file for skipped test must be empty: "{ref_path}"'
                    ),
                )
                return False
    return "skipped"


def refresh_runner_metadata() -> None:
    _refresh_registry()


refresh_runner_metadata()


SUITE_DEBUG_LOGGING = False


def set_suite_debug_logging(enabled: bool) -> None:
    global SUITE_DEBUG_LOGGING
    SUITE_DEBUG_LOGGING = enabled


def _log_suite_event(
    suite: SuiteInfo,
    *,
    event: Literal["setup", "teardown"],
    total: int,
) -> None:
    if not SUITE_DEBUG_LOGGING:
        return
    rel_dir = _relativize_path(suite.directory)
    action = "setting up" if event == "setup" else "tearing down"
    _CLI_LOGGER.debug("suite %s %s (%d tests) @ %s", action, suite.name, total, rel_dir)


class Worker:
    def __init__(
        self,
        queue: list[RunnerQueueItem],
        *,
        update: bool,
        coverage: bool = False,
        runner_versions: Mapping[str, str] | None = None,
        debug: bool = False,
        run_skipped_selector: RunSkippedSelector | None = None,
        jobs: int | None = None,
        test_slots: threading.Semaphore | None = None,
    ) -> None:
        self._queue = queue
        self._result: Summary | None = None
        self._exception: BaseException | None = None
        self._update = update
        self._coverage = coverage
        self._runner_versions = dict(runner_versions or {})
        self._debug = debug
        self._run_skipped_selector = run_skipped_selector or RunSkippedSelector()
        self._jobs = max(1, jobs if jobs is not None else get_default_jobs())
        self._test_slots = test_slots or threading.BoundedSemaphore(self._jobs)
        self._run_skipped_match_count = 0
        self._run_skipped_match_count_lock = threading.Lock()
        self._thread = threading.Thread(target=self._work)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> Summary:
        self._thread.join()
        if self._exception:
            raise self._exception
        if self._result is None:
            raise RuntimeError("worker finished without producing a result")
        return self._result

    @property
    def run_skipped_match_count(self) -> int:
        return self._run_skipped_match_count

    def _increment_run_skipped_match_count(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._run_skipped_match_count_lock:
            self._run_skipped_match_count += amount

    def _record_static_skip(
        self,
        *,
        test_item: TestQueueItem,
        fixtures: tuple[fixtures_impl.FixtureSpec, ...],
        reason: str,
        summary: Summary,
    ) -> None:
        handle_skip(
            reason,
            test_item.path,
            update=self._update,
            output_ext=getattr(test_item.runner, "output_ext", "txt"),
        )
        rel_path = _relativize_path(test_item.path)
        summary.total += 1
        summary.record_runner_outcome(test_item.runner.name, "skipped")
        if fixtures:
            summary.record_fixture_outcome(_fixture_names(fixtures), "skipped")
        summary.skipped += 1
        summary.skipped_paths.append(rel_path)

    def _skip_suite_with_reason(
        self,
        *,
        suite_item: SuiteQueueItem,
        tests: Sequence[TestQueueItem],
        displayed_reason: str,
        summary: Summary,
    ) -> bool:
        total = len(tests)
        if self._run_skipped_selector.should_run_skipped(reason=displayed_reason):
            self._increment_run_skipped_match_count(total)
            return False
        for skip_index, test_item in enumerate(tests, start=1):
            rel_path = _relativize_path(test_item.path)
            if is_verbose_output():
                with _push_suite_context(
                    name=suite_item.suite.name,
                    index=skip_index,
                    total=total,
                ):
                    suite_suffix = _format_suite_suffix()
                with stdout_lock:
                    print(f"{SKIP} skipped {rel_path}{suite_suffix}: {displayed_reason}")
            summary.total += 1
            summary.skipped += 1
            summary.skipped_paths.append(rel_path)
            summary.record_runner_outcome(test_item.runner.name, "skipped")
            summary.record_fixture_outcome(_fixture_names(suite_item.fixtures), "skipped")
        return True

    def _evaluate_suite_requirements(
        self,
        *,
        suite_item: SuiteQueueItem,
        tests: Sequence[TestQueueItem],
        requires: RequiresConfig,
        env: Mapping[str, str],
        config_args: Sequence[str],
    ) -> str | None:
        requirements = requires.as_mapping()
        missing_by_key: dict[str, set[str]] = {}
        unsupported_by_key: dict[str, set[str]] = {}
        for runner in {test_item.runner for test_item in tests}:
            result = runner.check_requirements(
                requirements,
                env=env,
                config_args=config_args,
            )
            for key in result.unsupported_keys:
                unsupported_by_key.setdefault(key, set()).add(runner.name)
            for key, required_values in result.missing_values.items():
                if not required_values:
                    continue
                missing_by_key.setdefault(key, set()).update(required_values)
        if unsupported_by_key:
            detail = ", ".join(
                f"{key} (runners: {', '.join(sorted(runners))})"
                for key, runners in sorted(unsupported_by_key.items())
            )
            raise HarnessError(
                f"suite '{suite_item.suite.name}' has unsupported requirement categories: {detail}"
            )
        if not missing_by_key:
            return None
        parts: list[str] = []
        for key, missing_set in sorted(missing_by_key.items()):
            rendered_values = ", ".join(sorted(missing_set))
            if key == "operators":
                parts.append(f"missing operators: {rendered_values}")
            else:
                parts.append(f"missing {key}: {rendered_values}")
        return "; ".join(parts)

    def _suite_static_skip_plan(
        self, suite_item: SuiteQueueItem
    ) -> list[tuple[TestQueueItem, str]] | None:
        planned: list[tuple[TestQueueItem, str]] = []
        for test_item in suite_item.tests:
            try:
                config = parse_test_config(test_item.path, coverage=self._coverage)
            except ValueError:
                return None
            skip_cfg = cast(SkipConfig | None, config.get("skip"))
            if skip_cfg is None or not skip_cfg.is_static or skip_cfg.reason is None:
                return None
            if self._run_skipped_selector.should_run_skipped(reason=skip_cfg.reason):
                return None
            planned.append((test_item, skip_cfg.reason))
        return planned

    def _work(self) -> Summary:
        try:
            self._result = Summary()
            result = self._result
            while True:
                if interrupt_requested():
                    break
                try:
                    queue_item = self._queue.pop()
                except IndexError:
                    break

                if isinstance(queue_item, SuiteQueueItem):
                    self._run_suite(queue_item, result)
                else:
                    self._run_test_item_with_slot(queue_item, result)
                if interrupt_requested():
                    break
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            self._exception = exc
            if self._result is None:
                self._result = Summary()
            return self._result

    def _run_test_item_with_slot(
        self,
        test_item: TestQueueItem,
        summary: Summary,
        *,
        suite_progress: tuple[str, int, int] | None = None,
        suite_fixtures: tuple[fixtures_impl.FixtureSpec, ...] | None = None,
        suite_assertion_lock: threading.Lock | None = None,
    ) -> bool:
        while not self._test_slots.acquire(timeout=0.1):
            if interrupt_requested():
                return True
        try:
            return self._run_test_item(
                test_item,
                summary,
                suite_progress=suite_progress,
                suite_fixtures=suite_fixtures,
                suite_assertion_lock=suite_assertion_lock,
            )
        finally:
            self._test_slots.release()

    def _run_suite_sequential(
        self,
        *,
        suite_item: SuiteQueueItem,
        tests: Sequence[TestQueueItem],
        summary: Summary,
    ) -> bool:
        total = len(tests)
        interrupted = False
        for index, test_item in enumerate(tests, start=1):
            if interrupt_requested():
                interrupted = True
                break
            interrupted = self._run_test_item_with_slot(
                test_item,
                summary,
                suite_progress=(suite_item.suite.name, index, total),
                suite_fixtures=suite_item.fixtures,
            )
            if interrupted:
                break
        return interrupted

    def _run_suite_member(
        self,
        *,
        run_context: contextvars.Context,
        test_item: TestQueueItem,
        suite_progress: tuple[str, int, int],
        suite_fixtures: tuple[fixtures_impl.FixtureSpec, ...],
        suite_assertion_lock: threading.Lock | None,
    ) -> tuple[Summary, bool]:
        local_summary = Summary()
        interrupted = run_context.run(
            self._run_test_item_with_slot,
            test_item,
            local_summary,
            suite_progress=suite_progress,
            suite_fixtures=suite_fixtures,
            suite_assertion_lock=suite_assertion_lock,
        )
        return local_summary, interrupted

    def _run_suite_parallel(
        self,
        *,
        suite_item: SuiteQueueItem,
        tests: Sequence[TestQueueItem],
        summary: Summary,
    ) -> bool:
        total = len(tests)
        max_workers = min(max(1, total), self._jobs)
        suite_assertion_lock = threading.Lock()
        suite_summary = Summary()
        interrupted = False
        futures: list[concurrent.futures.Future[tuple[Summary, bool]]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="suite",
        ) as executor:
            for index, test_item in enumerate(tests, start=1):
                if interrupt_requested():
                    interrupted = True
                    break
                run_context = contextvars.copy_context()
                futures.append(
                    executor.submit(
                        self._run_suite_member,
                        run_context=run_context,
                        test_item=test_item,
                        suite_progress=(suite_item.suite.name, index, total),
                        suite_fixtures=suite_item.fixtures,
                        suite_assertion_lock=suite_assertion_lock,
                    )
                )
            if interrupt_requested():
                interrupted = True
                for pending in futures:
                    if pending.done():
                        continue
                    pending.cancel()
            for future in concurrent.futures.as_completed(futures):
                if future.cancelled():
                    continue
                member_summary, member_interrupted = future.result()
                _merge_summary_inplace(suite_summary, member_summary)
                if member_interrupted:
                    interrupted = True
                    _request_interrupt()
                    for pending in futures:
                        if pending.done():
                            continue
                        pending.cancel()
        _merge_summary_inplace(summary, suite_summary)
        return interrupted

    def _run_suite(self, suite_item: SuiteQueueItem, summary: Summary) -> None:
        tests = suite_item.tests
        total = len(tests)
        if total == 0:
            return
        primary_test = tests[0].path
        try:
            primary_config = parse_test_config(primary_test, coverage=self._coverage)
        except ValueError as exc:
            raise RuntimeError(f"failed to parse suite config for {primary_test}: {exc}") from exc
        # Avoid activating suite fixtures when every suite member is statically skipped.
        static_skip_plan = self._suite_static_skip_plan(suite_item)
        if static_skip_plan is not None:
            for test_item, reason in static_skip_plan:
                self._record_static_skip(
                    test_item=test_item,
                    fixtures=suite_item.fixtures,
                    reason=reason,
                    summary=summary,
                )
            return
        inputs_override = typing.cast(str | None, primary_config.get("inputs"))
        env, config_args = get_test_env_and_config_args(primary_test, inputs=inputs_override)
        config_package_dirs = cast(tuple[str, ...], primary_config.get("package_dirs", tuple()))
        additional_package_dirs: list[str] = []
        for entry in config_package_dirs:
            additional_package_dirs.extend(_expand_package_dirs(Path(entry)))
        package_root = packages.find_package_root(primary_test)
        package_dir_candidates: list[str] = []
        if package_root is not None:
            env["TENZIR_PACKAGE_ROOT"] = str(package_root)
            if inputs_override is None:
                # Try nearest inputs/ directory, fall back to package-level
                nearest = _find_nearest_inputs_dir(primary_test, package_root)
                if nearest is not None:
                    env["TENZIR_INPUTS"] = str(nearest.resolve())
                else:
                    env["TENZIR_INPUTS"] = str((package_root / "tests" / "inputs"))
            package_dir_candidates.append(str(package_root))
        package_dir_candidates.extend(additional_package_dirs)
        for cli_path in _get_cli_packages():
            package_dir_candidates.extend(_expand_package_dirs(cli_path))
        if package_dir_candidates:
            merged_dirs = _deduplicate_package_dirs(package_dir_candidates)
            env["TENZIR_PACKAGE_DIRS"] = ",".join(merged_dirs)
            config_args = list(config_args) + [f"--package-dirs={','.join(merged_dirs)}"]
        skip_cfg = cast(SkipConfig | None, primary_config.get("skip"))
        requires_cfg = cast(RequiresConfig | None, primary_config.get("requires"))
        if isinstance(requires_cfg, RequiresConfig) and requires_cfg.has_requirements:
            missing_requirements = self._evaluate_suite_requirements(
                suite_item=suite_item,
                tests=tests,
                requires=requires_cfg,
                env=env,
                config_args=config_args,
            )
            if missing_requirements is not None:
                configured_reason = skip_cfg.reason if isinstance(skip_cfg, SkipConfig) else None
                reason = _compose_skip_reason(
                    configured_reason,
                    missing_requirements,
                    fallback=missing_requirements,
                )
                displayed_reason = f"capability unavailable: {reason}"
                allows_capability_skip = isinstance(
                    skip_cfg, SkipConfig
                ) and skip_cfg.allows_condition(SKIP_ON_CAPABILITY_UNAVAILABLE)
                if allows_capability_skip:
                    if not self._skip_suite_with_reason(
                        suite_item=suite_item,
                        tests=tests,
                        displayed_reason=displayed_reason,
                        summary=summary,
                    ):
                        raise HarnessError(displayed_reason)
                    return
                raise HarnessError(displayed_reason)
        suite_fixture_options = _build_fixture_options(suite_item.fixtures)
        context_token = fixtures_impl.push_context(
            fixtures_impl.FixtureContext(
                test=primary_test,
                config=typing.cast(dict[str, Any], primary_config),
                coverage=self._coverage,
                env=env,
                config_args=tuple(config_args),
                tenzir_binary=TENZIR_BINARY,
                tenzir_node_binary=TENZIR_NODE_BINARY,
                fixture_options=suite_fixture_options,
            )
        )
        _log_suite_event(suite_item.suite, event="setup", total=total)
        interrupted = False
        try:
            with fixtures_impl.suite_scope(suite_item.fixtures):
                if suite_item.suite.mode is SuiteExecutionMode.PARALLEL:
                    interrupted = self._run_suite_parallel(
                        suite_item=suite_item,
                        tests=tests,
                        summary=summary,
                    )
                else:
                    interrupted = self._run_suite_sequential(
                        suite_item=suite_item,
                        tests=tests,
                        summary=summary,
                    )
        except fixtures_impl.FixtureUnavailable as exc:
            if not (
                isinstance(skip_cfg, SkipConfig)
                and skip_cfg.allows_condition(SKIP_ON_FIXTURE_UNAVAILABLE)
            ):
                raise
            reason = _compose_skip_reason(skip_cfg.reason, str(exc))
            displayed_reason = f"fixture unavailable: {reason}"
            if not self._skip_suite_with_reason(
                suite_item=suite_item,
                tests=tests,
                displayed_reason=displayed_reason,
                summary=summary,
            ):
                raise
            _CLI_LOGGER.warning(
                "fixture unavailable for suite '%s': %s",
                suite_item.suite.name,
                reason,
            )
        finally:
            _log_suite_event(suite_item.suite, event="teardown", total=total)
            fixtures_impl.pop_context(context_token)
            cleanup_test_tmp_dir(env.get(TEST_TMP_ENV_VAR))
        if interrupted:
            _request_interrupt()

    def _run_test_item(
        self,
        test_item: TestQueueItem,
        summary: Summary,
        *,
        suite_progress: tuple[str, int, int] | None = None,
        suite_fixtures: tuple[fixtures_impl.FixtureSpec, ...] | None = None,
        suite_assertion_lock: threading.Lock | None = None,
    ) -> bool:
        test_path = test_item.path
        runner = test_item.runner
        rel_path = _relativize_path(test_path)
        if interrupt_requested():
            return True
        configured_fixtures = suite_fixtures or _get_test_fixtures(
            test_path, coverage=self._coverage
        )
        fixtures = configured_fixtures
        retry_limit = 1
        config: TestConfig | None = None
        fixture_assertions: dict[str, Any] = {}
        parse_error: str | None = None
        try:
            config = parse_test_config(test_path, coverage=self._coverage)
        except ValueError as exc:
            parse_error = str(exc)
            config = None
        if config is not None:
            raw_retry = config.get("retry", 0)
            if isinstance(raw_retry, int):
                retry_limit = max(1, raw_retry)
            config_fixtures = cast(
                tuple[fixtures_impl.FixtureSpec, ...], config.get("fixtures", tuple())
            )
            if suite_fixtures is not None and config_fixtures != suite_fixtures:
                raise RuntimeError(
                    f"fixture mismatch for suite test {test_path}: "
                    f"expected {suite_fixtures}, got {config_fixtures}"
                )
            if suite_fixtures is None:
                configured_fixtures = config_fixtures
            try:
                fixture_assertions = _build_fixture_assertions(
                    cast(
                        Mapping[str, Mapping[str, Any]] | None,
                        config.get("assertions"),
                    )
                )
            except ValueError as exc:
                parse_error = str(exc)
            skip_cfg = cast(SkipConfig | None, config.get("skip"))
            if skip_cfg is not None and skip_cfg.is_static:
                assert skip_cfg.reason is not None
                if self._run_skipped_selector.should_run_skipped(reason=skip_cfg.reason):
                    self._increment_run_skipped_match_count()
                else:
                    self._record_static_skip(
                        test_item=test_item,
                        fixtures=fixtures,
                        reason=skip_cfg.reason,
                        summary=summary,
                    )
                    return False
            # Conditional skips are suite-level — _run_suite handles them.
        if parse_error is not None:
            message = format_failure_message(parse_error)
            report_failure(test_path, message)
            summary.total += 1
            summary.record_runner_outcome(runner.name, False)
            if fixtures:
                summary.record_fixture_outcome(_fixture_names(fixtures), False)
            summary.failed += 1
            summary.failed_paths.append(rel_path)
            return False
        detail_bits = [f"runner={runner.name}"]
        if fixtures:
            detail_bits.append(f"fixtures={', '.join(str(s) for s in fixtures)}")
        if suite_progress:
            name, index, total = suite_progress
            detail_bits.append(f"suite={name} ({index}/{total})")
        detail_segment = f" ({', '.join(detail_bits)})" if detail_bits else ""
        if is_passthrough_enabled():
            with stdout_lock:
                print(f"{INFO} running {rel_path}{detail_segment} [passthrough]")
        elif self._debug:
            _CLI_LOGGER.debug("running %s%s", rel_path, detail_segment)
        max_attempts = retry_limit
        attempts = 0
        final_outcome: bool | str = False
        final_interrupted = False
        while attempts < max_attempts:
            if interrupt_requested():
                final_interrupted = True
                break
            attempts += 1
            with _push_retry_context(attempt=attempts, max_attempts=max_attempts):
                with _push_assertion_check_context() as assertion_tally:
                    attempt_context = contextlib.ExitStack()
                    if suite_progress is not None:
                        name, index, total = suite_progress
                        attempt_context.enter_context(
                            _push_suite_context(name=name, index=index, total=total)
                        )
                    if suite_fixtures is not None and config is not None:
                        attempt_context.enter_context(
                            _push_fixture_test_context(
                                test=test_path,
                                config=config,
                                fixture_assertions=fixture_assertions,
                            )
                        )
                    interrupted = False
                    try:
                        with attempt_context:
                            outcome = runner.run(test_path, self._update, self._coverage)
                            if outcome and outcome != "skipped" and suite_fixtures is not None:
                                if suite_assertion_lock is None:
                                    _run_fixture_assertions_for_test(
                                        test=test_path,
                                        fixture_specs=suite_fixtures,
                                        fixture_assertions=fixture_assertions,
                                    )
                                else:
                                    with suite_assertion_lock:
                                        _run_fixture_assertions_for_test(
                                            test=test_path,
                                            fixture_specs=suite_fixtures,
                                            fixture_assertions=fixture_assertions,
                                        )
                    except KeyboardInterrupt:  # pragma: no cover - defensive guard
                        _request_interrupt()
                        interrupted = True
                        outcome = False
                    except Exception as exc:
                        is_fixture_assertion = str(exc).startswith("fixture assertion failed:")
                        error_message = (
                            _fixture_assertion_failure_message(exc)
                            if is_fixture_assertion
                            else format_failure_message(str(exc))
                        )
                        report_failure(test_path, error_message)
                        outcome = False
                        final_interrupted = False
                        final_outcome = outcome
                        if is_fixture_assertion and attempts < max_attempts:
                            summary.record_assertion_checks(
                                total=assertion_tally.total,
                                failed=assertion_tally.failed,
                            )
                            continue
                        summary.record_assertion_checks(
                            total=assertion_tally.total,
                            failed=assertion_tally.failed,
                        )
                        break
                    summary.record_assertion_checks(
                        total=assertion_tally.total,
                        failed=assertion_tally.failed,
                    )

                if interrupted:
                    report_failure(test_path, _INTERRUPTED_NOTICE)
                    final_interrupted = True
                final_outcome = outcome
                if final_interrupted or interrupt_requested():
                    break
                if outcome == "skipped" or outcome:
                    break
                if attempts < max_attempts:
                    continue
        if attempts == 0 and final_interrupted:
            # The test never started an execution attempt (for example, work
            # was cancelled after an interrupt), so do not count it as failed.
            return True

        summary.total += 1
        summary.record_runner_outcome(runner.name, final_outcome)
        if fixtures:
            summary.record_fixture_outcome(_fixture_names(fixtures), final_outcome)
        if final_outcome == "skipped":
            summary.skipped += 1
            summary.skipped_paths.append(rel_path)
        elif not final_outcome:
            summary.failed += 1
            summary.failed_paths.append(rel_path)
        return final_interrupted or interrupt_requested()


def get_runner_for_test(test_path: Path) -> Runner:
    """Determine the appropriate runner for a test based on its configuration."""
    return runners_get_runner(test_path)


def collect_all_tests(directory: Path) -> Iterator[Path]:
    if directory.name in {"fixtures", "runners"}:
        return
    extensions = _allowed_extensions or {
        ext
        for runner in runners_iter_runners()
        if (ext := getattr(runner, "_ext", None)) is not None
    }
    for ext in extensions:
        for candidate in directory.glob(f"**/*.{ext}"):
            if _is_inputs_path(candidate):
                continue
            yield candidate


_FNMATCH_META = set("*?[")


def _normalize_pattern(pattern: str) -> str:
    """Auto-wrap bare strings in ``*…*`` so they act as substring matches.

    Patterns that already contain fnmatch metacharacters (``*``, ``?``, ``[``)
    are returned unchanged.
    """
    if not _FNMATCH_META.intersection(pattern):
        return f"*{pattern}*"
    return pattern


def _filter_paths_by_patterns(
    paths: Iterable[Path],
    patterns: Sequence[str],
    *,
    project_root: Path,
) -> set[Path]:
    """Return the subset of *paths* whose relative POSIX name matches any pattern.

    Paths are resolved and converted to forward-slash (POSIX) format relative
    to *project_root* before matching.  Matching is case-sensitive
    (``fnmatch.fnmatchcase``).  Empty or whitespace-only patterns are silently
    skipped.  A path is included if it matches **any** non-empty pattern (OR
    semantics).

    Bare strings without glob metacharacters (``*``, ``?``, ``[``) are
    automatically treated as substring matches by wrapping them in ``*…*``.
    """
    matched: set[Path] = set()
    resolved_root = project_root.resolve()
    for path in paths:
        try:
            rel = path.resolve().relative_to(resolved_root).as_posix()
        except ValueError:
            _CLI_LOGGER.debug("skipping path %s outside project root %s", path, resolved_root)
            continue
        for pattern in patterns:
            if not pattern or not pattern.strip():
                continue
            if fnmatch.fnmatchcase(rel, _normalize_pattern(pattern)):
                matched.add(path)
                break
    return matched


def _expand_suites(paths: set[Path]) -> set[Path]:
    """If any test in a suite was selected, include all tests from that suite."""
    suite_dirs: set[Path] = set()
    for path in paths:
        suite_info = _resolve_suite_for_test(path)
        if suite_info is not None:
            suite_dirs.add(suite_info.directory)
    if not suite_dirs:
        return paths
    expanded = set(paths)
    for suite_dir in suite_dirs:
        for test_path in collect_all_tests(suite_dir):
            expanded.add(test_path.resolve())
    return expanded


def _configure_runtime_logging(*, debug_enabled: bool) -> None:
    """Apply CLI/runtime logger configuration for test and fixture execution paths."""

    set_debug_logging(debug_enabled)
    _set_discovery_logging(debug_enabled)
    set_suite_debug_logging(debug_enabled)

    fixture_logger = logging.getLogger("tenzir_test.fixtures")
    root_logger = logging.getLogger()

    debug_formatter = logging.Formatter(f"{DEBUG_PREFIX} %(message)s")
    default_formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    if debug_enabled:
        if not root_logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(debug_formatter)
            root_logger.addHandler(stream_handler)
        else:
            for existing_handler in list(root_logger.handlers):
                existing_handler.setFormatter(debug_formatter)
        root_logger.setLevel(logging.INFO)
        if not fixture_logger.handlers:
            fixture_logger.addHandler(logging.StreamHandler())
        for handler in list(fixture_logger.handlers):
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(debug_formatter)
        fixture_logger.setLevel(logging.DEBUG)
        fixture_logger.propagate = False
        return

    for existing_handler in list(root_logger.handlers):
        existing_handler.setFormatter(default_formatter)
    root_logger.setLevel(logging.WARNING)
    for handler in list(fixture_logger.handlers):
        handler.setLevel(logging.INFO)
        handler.setFormatter(default_formatter)
    fixture_logger.setLevel(logging.WARNING)
    fixture_logger.propagate = True


def _print_fixture_env_lines(env: Mapping[str, str]) -> None:
    """Print fixture-provided environment variables in deterministic order."""

    with stdout_lock:
        for key in sorted(env.keys()):
            print(f"{key}={env[key]}")
        print(f"{INFO} fixture environment is active; press Ctrl+C to stop")


def _wait_for_fixture_shutdown() -> None:
    """Block until SIGINT/SIGTERM is received, then return."""

    stop_event = threading.Event()
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def _handle_signal(signum: int, frame: object | None) -> None:  # pragma: no cover - signal path
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        while not stop_event.wait(0.25):
            continue
    except KeyboardInterrupt:  # pragma: no cover - defensive fallback
        pass
    finally:
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])


def _raise_fixture_unavailable_harness_error(exc: fixtures_impl.FixtureUnavailable) -> None:
    """Raise a sanitized HarnessError for an unavailable fixture."""

    reason = _sanitize_message(str(exc))
    if reason:
        raise HarnessError(f"fixture unavailable: {reason}") from exc
    raise HarnessError("fixture unavailable") from exc


def run_fixture_mode_cli(
    *,
    root: Path | None,
    package_dirs: Sequence[Path] | None,
    fixtures: Sequence[str],
    debug: bool,
    keep_tmp_dirs: bool,
) -> int:
    """Activate requested fixtures in foreground mode until interrupted."""

    if not fixtures:
        raise HarnessError("error: at least one --fixture option is required")

    debug_enabled = bool(debug or _default_debug_logging)
    _configure_runtime_logging(debug_enabled=debug_enabled)
    set_keep_tmp_dirs(bool(os.environ.get(_TMP_KEEP_ENV_VAR)) or keep_tmp_dirs)

    settings = discover_settings(root=root)
    apply_settings(settings)
    _set_cli_packages(list(package_dirs or []))

    try:
        _load_project_fixtures(ROOT, expose_namespace=True)
    except RuntimeError as exc:
        raise HarnessError(f"error: {exc}") from exc

    try:
        fixture_specs = _normalize_cli_fixture_specs(fixtures)
    except ValueError as exc:
        raise HarnessError(f"error: {exc}") from exc

    fixture_test = ROOT / ".fixture-mode"
    env, config_args = get_test_env_and_config_args(fixture_test)

    package_dir_candidates: list[str] = []
    package_root = packages.find_package_root(fixture_test)
    if package_root is not None:
        env["TENZIR_PACKAGE_ROOT"] = str(package_root)
        package_dir_candidates.append(str(package_root))
    for cli_path in _get_cli_packages():
        package_dir_candidates.extend(_expand_package_dirs(cli_path))
    if package_dir_candidates:
        merged_dirs = _deduplicate_package_dirs(package_dir_candidates)
        env["TENZIR_PACKAGE_DIRS"] = ",".join(merged_dirs)
        config_args.append(f"--package-dirs={','.join(merged_dirs)}")

    for spec in fixture_specs:
        try:
            fixtures_impl.acquire_fixture(spec.name)
        except ValueError as exc:
            raise HarnessError(f"error: {exc}") from exc

    try:
        fixture_options = _build_fixture_options(fixture_specs)
    except ValueError as exc:
        raise HarnessError(f"error: {exc}") from exc
    context_token = fixtures_impl.push_context(
        fixtures_impl.FixtureContext(
            test=fixture_test,
            config={"fixtures": fixture_specs},
            coverage=False,
            env=env,
            config_args=tuple(config_args),
            tenzir_binary=TENZIR_BINARY,
            tenzir_node_binary=TENZIR_NODE_BINARY,
            fixture_options=fixture_options,
        )
    )
    try:
        with fixtures_impl.activate(fixture_specs) as fixture_env:
            env.update(fixture_env)
            _apply_fixture_env(env, fixture_specs)
            _print_fixture_env_lines(fixture_env)
            _wait_for_fixture_shutdown()
    except fixtures_impl.FixtureUnavailable as exc:
        _raise_fixture_unavailable_harness_error(exc)
    finally:
        fixtures_impl.pop_context(context_token)
        cleanup_test_tmp_dir(env.get(TEST_TMP_ENV_VAR))

    return 0


def run_cli(
    *,
    root: Path | None,
    package_dirs: Sequence[Path] | None = None,
    tests: Sequence[Path],
    update: bool,
    debug: bool,
    purge: bool,
    coverage: bool,
    coverage_source_dir: Path | None,
    runner_summary: bool,
    fixture_summary: bool,
    show_summary: bool,
    show_diff_output: bool,
    show_diff_stat: bool,
    jobs: int,
    keep_tmp_dirs: bool,
    passthrough: bool,
    verbose: bool = False,
    run_skipped: bool = False,
    run_skipped_reasons: Sequence[str] = (),
    jobs_overridden: bool = False,
    all_projects: bool = False,
    match_patterns: Sequence[str] = (),
) -> ExecutionResult:
    """Execute the harness and return a structured result for library consumers.

    Args:
        verbose: Print individual test results (pass/skip) as they complete.
            When False (default), only failures are printed during execution.
        run_skipped: When True, ignore all skip configuration and execute
            skipped tests unconditionally.
        run_skipped_reasons: Repeatable reason patterns for selectively
            running skipped tests. Matching uses the same substring/glob
            semantics as ``--match``.
        match_patterns: Substring or glob patterns matched against relative
            test paths.  Bare strings without glob metacharacters (``*``,
            ``?``, ``[``) are treated as substring matches.  Tests matching
            any pattern are selected.  When path arguments are also provided
            via *tests*, only tests matching both are run (intersection).
            Empty or whitespace-only patterns are silently ignored.
    """
    from tenzir_test.engine import state as engine_state

    try:
        debug_enabled = bool(debug or _default_debug_logging)
        _configure_runtime_logging(debug_enabled=debug_enabled)

        set_keep_tmp_dirs(bool(os.environ.get(_TMP_KEEP_ENV_VAR)) or keep_tmp_dirs)
        set_show_diff_output(show_diff_output)
        set_show_diff_stat(show_diff_stat)
        if passthrough:
            harness_mode = HarnessMode.PASSTHROUGH
        elif update:
            harness_mode = HarnessMode.UPDATE
        else:
            harness_mode = HarnessMode.COMPARE
        set_harness_mode(harness_mode)
        passthrough_mode = harness_mode is HarnessMode.PASSTHROUGH
        # Passthrough mode requires verbose output to show real-time test results
        # Debug mode also implies verbose to show all test results when diagnosing
        set_verbose_output(verbose or passthrough_mode or debug_enabled)
        if passthrough_mode and jobs > 1:
            if jobs_overridden:
                print(f"{INFO} forcing --jobs=1 in passthrough mode to preserve output ordering")
            jobs = 1
        if passthrough_mode and update:
            print(f"{INFO} ignoring --update in passthrough mode")
            update = False

        settings = discover_settings(root=root)
        apply_settings(settings)
        _set_cli_packages(list(package_dirs or []))
        selected_tests = list(tests)
        run_skipped_selector = RunSkippedSelector.from_cli(
            run_all=run_skipped,
            reason_patterns=run_skipped_reasons,
        )

        plan: ExecutionPlan | None = None
        if selected_tests:
            plan = _build_execution_plan(
                ROOT,
                selected_tests,
                root_explicit=root is not None,
                all_projects=all_projects,
            )

        if not _is_project_root(ROOT):
            if all_projects:
                raise HarnessError(
                    "error: --all-projects requires a project root; specify one with --root"
                )
            if not selected_tests:
                message = (
                    f"{INFO} no tenzir-test project detected at {ROOT}.\n"
                    f"{INFO} run from your project root or provide --root."
                )
                print(message)
                raise HarnessError(message, show_message=False)
            assert plan is not None
            runnable_satellites = [item for item in plan.satellites if item.should_run()]
            if not runnable_satellites:
                message = (
                    f"{INFO} no tenzir-test project detected at {ROOT}.\n"
                    f"{INFO} run from your project root or provide --root."
                )
                print(message)
                sample = ", ".join(str(path) for path in selected_tests[:3])
                if len(selected_tests) > 3:
                    sample += ", ..."
                print(f"{INFO} ignoring provided selection(s): {sample}")
                raise HarnessError(
                    "no runnable tests selected outside of a project root", show_message=False
                )
        if plan is None:
            plan = _build_execution_plan(
                ROOT,
                selected_tests,
                root_explicit=root is not None,
                all_projects=all_projects,
            )
        display_base = Path.cwd().resolve()
        project_count = _print_execution_plan(plan, display_base=display_base)
        if project_count:
            print()

        with _install_interrupt_handler():
            engine_state.refresh()

            overall_summary = Summary()
            overall_queue_count = 0
            executed_projects: list[ProjectSelection] = []
            project_results: list[ProjectResult] = []
            printed_projects = 0
            interrupted = False

            for selection in plan.projects():
                if interrupt_requested():
                    break
                if not selection.should_run():
                    if selection.kind == "root":
                        _set_project_root(selection.root)
                        engine_state.refresh()
                        try:
                            _load_project_runners(selection.root, expose_namespace=True)
                            _load_project_fixtures(selection.root, expose_namespace=True)
                        except RuntimeError as exc:
                            raise HarnessError(f"error: {exc}") from exc
                        refresh_runner_metadata()
                        _set_project_root(settings.root)
                        engine_state.refresh()
                    continue

                if printed_projects:
                    print()

                _set_project_root(selection.root)
                engine_state.refresh()
                expose_namespace = selection.kind == "root"
                try:
                    _load_project_runners(selection.root, expose_namespace=expose_namespace)
                    _load_project_fixtures(selection.root, expose_namespace=expose_namespace)
                except RuntimeError as exc:
                    raise HarnessError(f"error: {exc}") from exc
                refresh_runner_metadata()

                tests_to_run = selection.selectors if not selection.run_all else [selection.root]

                if purge:
                    continue

                collected_paths: set[Path] = set()
                for test in tests_to_run:
                    if test.resolve() == selection.root.resolve():
                        all_tests = []
                        for tests_dir in _iter_project_test_directories(selection.root):
                            all_tests.extend(list(collect_all_tests(tests_dir)))
                        for test_path in all_tests:
                            collected_paths.add(test_path.resolve())
                        continue

                    resolved = test.resolve()
                    if not resolved.exists():
                        raise HarnessError(f"error: test path `{test}` does not exist")

                    if resolved.is_dir():
                        if _is_inputs_path(resolved):
                            continue
                        tql_files = list(collect_all_tests(resolved))
                        if not tql_files:
                            raise HarnessError(
                                f"error: no {_allowed_extensions} files found in {resolved}"
                            )
                        for file_path in tql_files:
                            suite_info = _resolve_suite_for_test(file_path)
                            if suite_info is None:
                                continue
                            suite_dir = suite_info.directory
                            if resolved == suite_dir:
                                continue
                            if _path_is_within(resolved, suite_dir):
                                rel_target = _relativize_path(resolved)
                                rel_suite = _relativize_path(suite_dir)
                                detail = (
                                    f"cannot select {rel_target} directly because it is inside the suite "
                                    f"'{suite_info.name}' defined in {rel_suite / _CONFIG_FILE_NAME}."
                                )
                                print(f"{CROSS} {detail}", file=sys.stderr)
                                print(
                                    f"{INFO} select the suite directory instead",
                                    file=sys.stderr,
                                )
                                raise HarnessError(
                                    f"invalid partial suite selection at {rel_target}",
                                    show_message=False,
                                )
                        for file_path in tql_files:
                            collected_paths.add(file_path.resolve())
                    elif resolved.is_file():
                        if _is_inputs_path(resolved):
                            continue
                        if resolved.suffix[1:] in _allowed_extensions:
                            suite_info = _resolve_suite_for_test(resolved)
                            if suite_info is not None and _path_is_within(
                                resolved, suite_info.directory
                            ):
                                rel_file = _relativize_path(resolved)
                                rel_suite = _relativize_path(suite_info.directory)
                                detail = (
                                    f"cannot select {rel_file} directly because it belongs to the suite "
                                    f"'{suite_info.name}' defined in {rel_suite / _CONFIG_FILE_NAME}."
                                )
                                print(f"{CROSS} {detail}", file=sys.stderr)
                                print(f"{INFO} select the suite directory instead", file=sys.stderr)
                                raise HarnessError(
                                    f"invalid suite selection for {rel_file}",
                                    show_message=False,
                                )
                            collected_paths.add(resolved.resolve())
                        else:
                            raise HarnessError(
                                f"error: unsupported file type {resolved.suffix} for {resolved} - only {_allowed_extensions} files are supported"
                            )
                    else:
                        raise HarnessError(f"error: `{test}` is neither a file nor a directory")

                active_patterns = [p.strip() for p in match_patterns if p and p.strip()]
                if active_patterns:
                    # Filter collected tests to those matching any
                    # pattern.  This acts as an intersection when path
                    # args were given (since collected_paths was
                    # pre-populated above), or filters all discovered
                    # tests when no paths were specified.
                    collected_paths = _filter_paths_by_patterns(
                        collected_paths,
                        active_patterns,
                        project_root=selection.root,
                    )
                    collected_paths = _expand_suites(collected_paths)
                    if not collected_paths:
                        pattern_list = ", ".join(repr(p) for p in active_patterns)
                        if not selection.run_all:
                            print(
                                f"{INFO} no tests matched pattern(s) {pattern_list}"
                                f" within the selected paths"
                            )
                        else:
                            print(f"{INFO} no tests matched pattern(s): {pattern_list}")

                if interrupt_requested():
                    break

                queue = _build_queue_from_paths(collected_paths, coverage=coverage)
                queue.sort(key=_queue_sort_key, reverse=True)
                project_queue_size = _count_queue_tests(queue)
                project_summary = Summary()
                job_count, enabled_flags, verb = _summarize_harness_configuration(
                    jobs=jobs,
                    update=update,
                    coverage=coverage,
                    debug=debug_enabled,
                    show_summary=show_summary,
                    runner_summary=runner_summary,
                    fixture_summary=fixture_summary,
                    passthrough=passthrough_mode,
                    run_skipped_selector=run_skipped_selector,
                )

                if not project_queue_size:
                    overall_queue_count += project_queue_size
                    executed_projects.append(selection)
                    project_results.append(
                        ProjectResult(
                            selection=selection,
                            summary=project_summary,
                            queue_size=project_queue_size,
                        )
                    )
                    continue

                os.environ["TENZIR_EXEC__DUMP_DIAGNOSTICS"] = "true"
                if not TENZIR_BINARY:
                    raise HarnessError(
                        f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`"
                    )
                try:
                    tenzir_version = get_version()
                except FileNotFoundError:
                    raise HarnessError(
                        f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`"
                    )
                except subprocess.CalledProcessError as exc:
                    if _is_interrupt_exit(exc.returncode) or interrupt_requested():
                        _request_interrupt()
                        interrupted = True
                        break
                    raise

                runner_versions = _collect_runner_versions(queue, tenzir_version=tenzir_version)
                runner_breakdown = _runner_breakdown(
                    queue,
                    tenzir_version=tenzir_version,
                    runner_versions=runner_versions,
                )

                _print_project_start(
                    selection=selection,
                    display_base=display_base,
                    queue_size=project_queue_size,
                    job_count=job_count,
                    enabled_flags=enabled_flags,
                    verb=verb,
                )
                count_width = max((len(str(count)) for _, count, _ in runner_breakdown), default=1)
                for name, count, version in runner_breakdown:
                    version_segment = f" (v{version})" if version else ""
                    print(f"{INFO}   {count:>{count_width}}× {name}{version_segment}")
                printed_projects += 1

                test_slots = threading.BoundedSemaphore(max(1, jobs))
                workers = [
                    Worker(
                        queue,
                        update=update,
                        coverage=coverage,
                        runner_versions=runner_versions,
                        debug=debug_enabled,
                        run_skipped_selector=run_skipped_selector,
                        jobs=jobs,
                        test_slots=test_slots,
                    )
                    for _ in range(jobs)
                ]
                for worker in workers:
                    worker.start()
                skip_filter_matches = 0
                try:
                    for worker in workers:
                        project_summary += worker.join()
                        skip_filter_matches += worker.run_skipped_match_count
                except KeyboardInterrupt:  # pragma: no cover - defensive guard
                    _request_interrupt()
                    for worker in workers:
                        worker.join()
                    interrupted = True
                    break
                except fixtures_impl.FixtureUnavailable as exc:
                    _raise_fixture_unavailable_harness_error(exc)
                except Exception as exc:
                    if not (_exception_is_interrupt(exc) or interrupt_requested()):
                        raise
                    _request_interrupt()
                    for remaining_worker in workers:
                        try:
                            remaining_worker.join()
                        except Exception as join_exc:
                            if not (_exception_is_interrupt(join_exc) or interrupt_requested()):
                                raise
                    interrupted = True
                    break
                if run_skipped_selector.has_reason_filters and skip_filter_matches == 0:
                    print(f"{INFO} no skipped tests matched run-skipped filters")

                _print_compact_summary(project_summary)
                summary_enabled = show_summary or runner_summary or fixture_summary
                if summary_enabled:
                    if is_verbose_output():
                        _print_detailed_summary(project_summary)
                    _print_ascii_summary(
                        project_summary,
                        include_runner=runner_summary,
                        include_fixture=fixture_summary,
                    )

                if coverage:
                    coverage_dir = os.environ.get(
                        "CMAKE_COVERAGE_OUTPUT_DIRECTORY", os.path.join(os.getcwd(), "coverage")
                    )
                    source_dir = str(coverage_source_dir) if coverage_source_dir else os.getcwd()
                    print(f"{INFO} Code coverage data collected in {coverage_dir}")
                    print(f"{INFO} Source directory for coverage mapping: {source_dir}")

                overall_summary += project_summary
                overall_queue_count += project_queue_size
                executed_projects.append(selection)
                project_results.append(
                    ProjectResult(
                        selection=selection,
                        summary=project_summary,
                        queue_size=project_queue_size,
                    )
                )

                if interrupt_requested():
                    break

            # Restore root project context for subsequent operations.
            _set_project_root(settings.root)
            engine_state.refresh()

            interrupted = interrupted or interrupt_requested()
            if interrupted:
                return ExecutionResult(
                    summary=overall_summary,
                    project_results=tuple(project_results),
                    queue_size=overall_queue_count,
                    exit_code=130,
                    interrupted=True,
                )

            if purge:
                for runner in runners_iter_runners():
                    runner.purge()
                return ExecutionResult(
                    summary=overall_summary,
                    project_results=tuple(project_results),
                    queue_size=overall_queue_count,
                    exit_code=0,
                    interrupted=interrupted,
                )

            if overall_queue_count == 0:
                print(f"{INFO} no tests selected")
                return ExecutionResult(
                    summary=overall_summary,
                    project_results=tuple(project_results),
                    queue_size=overall_queue_count,
                    exit_code=0,
                    interrupted=interrupted,
                )

            if len(executed_projects) > 1:
                _print_aggregate_totals(len(executed_projects), overall_summary)

            exit_code = 1 if overall_summary.failed > 0 else 0
            return ExecutionResult(
                summary=overall_summary,
                project_results=tuple(project_results),
                queue_size=overall_queue_count,
                exit_code=exit_code,
                interrupted=False,
            )

    finally:
        _cleanup_all_tmp_dirs()


def execute(
    *,
    root: Path | None = None,
    tests: Sequence[Path] = (),
    update: bool = False,
    debug: bool = False,
    purge: bool = False,
    coverage: bool = False,
    coverage_source_dir: Path | None = None,
    runner_summary: bool = False,
    fixture_summary: bool = False,
    show_summary: bool = False,
    show_diff_output: bool = True,
    show_diff_stat: bool = True,
    jobs: int | None = None,
    keep_tmp_dirs: bool = False,
    passthrough: bool = False,
    verbose: bool = False,
    run_skipped: bool = False,
    run_skipped_reasons: Sequence[str] = (),
    jobs_overridden: bool = False,
    all_projects: bool = False,
    match_patterns: Sequence[str] = (),
) -> ExecutionResult:
    """Library-oriented wrapper around `run_cli` with defaulted parameters.

    Args:
        verbose: Print individual test results (pass/skip) as they complete.
            When False (default), only failures are printed during execution.
        run_skipped: When True, ignore all skip configuration and execute
            skipped tests unconditionally.
        run_skipped_reasons: Repeatable reason patterns for selectively
            running skipped tests. Matching uses the same substring/glob
            semantics as ``--match``.
        match_patterns: Substring or glob patterns matched against relative
            test paths.  Bare strings without glob metacharacters (``*``,
            ``?``, ``[``) are treated as substring matches.  Tests matching
            any pattern are selected.  When path arguments are also provided
            via *tests*, only tests matching both are run (intersection).
            Empty or whitespace-only patterns are silently ignored.
    """
    resolved_jobs = jobs if jobs is not None else get_default_jobs()
    return run_cli(
        root=root,
        tests=list(tests),
        update=update,
        debug=debug,
        purge=purge,
        coverage=coverage,
        coverage_source_dir=coverage_source_dir,
        runner_summary=runner_summary,
        fixture_summary=fixture_summary,
        show_summary=show_summary,
        show_diff_output=show_diff_output,
        show_diff_stat=show_diff_stat,
        jobs=resolved_jobs,
        keep_tmp_dirs=keep_tmp_dirs,
        passthrough=passthrough,
        verbose=verbose,
        run_skipped=run_skipped,
        run_skipped_reasons=run_skipped_reasons,
        jobs_overridden=jobs_overridden,
        all_projects=all_projects,
        match_patterns=match_patterns,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import click

    from . import cli as cli_module

    try:
        result = cli_module.cli.main(
            args=list(argv) if argv is not None else None,
            standalone_mode=False,
        )
    except click.exceptions.ClickException as exc:
        exc.show(file=sys.stderr)
        exit_code = getattr(exc, "exit_code", 1)
        raise SystemExit(exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Abort as exc:
        raise SystemExit(1) from exc
    exit_code = cli_module._normalize_exit_code(result)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


def _print_project_start(
    *,
    selection: ProjectSelection,
    display_base: Path,
    queue_size: int,
    job_count: int,
    enabled_flags: str,
    verb: str,
) -> None:
    project_name = selection.root.name or selection.root.as_posix()
    if selection.kind == "root":
        project_kind = "root project"
    elif packages.is_package_dir(selection.root):
        project_kind = "package project"
    else:
        project_kind = "satellite project"

    location = _format_relative_path(selection.root, display_base)
    if location != "." and not location.startswith(("./", "../")):
        location_display = f"./{location}"
    else:
        location_display = location
    project_display = f"{BOLD}{project_name}{RESET_COLOR}"
    toggles = f"; {enabled_flags}" if enabled_flags else ""
    jobs_segment = f" ({job_count} jobs)" if job_count else ""
    print(
        f"{INFO} {project_display}: {verb} {queue_size} tests{jobs_segment} from {project_kind} at {location_display}{toggles}"
    )
