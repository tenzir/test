#!/usr/bin/env python3


from __future__ import annotations

import atexit
import builtins
import contextlib
import dataclasses
import difflib
import enum
import importlib.util
import logging
import os
import re
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

RunnerQueueItem = tuple[Runner, Path]


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


class HarnessMode(enum.Enum):
    """Internal execution modes for the harness."""

    COMPARE = "compare"
    UPDATE = "update"
    PASSTHROUGH = "passthrough"


def detect_execution_mode(root: Path) -> tuple[ExecutionMode, Path | None]:
    """Return the execution mode and detected package root for `root`."""

    if packages.is_package_dir(root):
        return ExecutionMode.PACKAGE, root

    parent = root.parent if root.name == "tests" else None
    if parent is not None and packages.is_package_dir(parent):
        return ExecutionMode.PACKAGE, parent

    return ExecutionMode.PROJECT, None


_settings = discover_settings()
TENZIR_BINARY = _settings.tenzir_binary
TENZIR_NODE_BINARY = _settings.tenzir_node_binary
ROOT = _settings.root
INPUTS_DIR = _settings.inputs_dir
EXECUTION_MODE, _DETECTED_PACKAGE_ROOT = detect_execution_mode(ROOT)
HARNESS_MODE = HarnessMode.COMPARE
CHECKMARK = "\033[92;1m✔\033[0m"
CROSS = "\033[31m✘\033[0m"
INFO = "\033[94;1mi\033[0m"
SKIP = "\033[90;1m●\033[0m"
VERBOSE_PREFIX = "\033[94;1m▶\033[0m"
DEBUG_PREFIX = "\033[95m◆\033[0m"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

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


_INTERRUPTED_NOTICE = "└─▶ \033[33mtest interrupted by user\033[0m"

_CURRENT_TEST_CONTEXT = threading.local()
SHOW_TEST_DETAILS = False


@contextlib.contextmanager
def _push_test_context(
    *, runner_name: str | None, runner_version: str | None, fixtures: tuple[str, ...]
) -> Iterator[None]:
    previous = getattr(_CURRENT_TEST_CONTEXT, "value", None)
    _CURRENT_TEST_CONTEXT.value = {
        "runner_name": runner_name,
        "runner_version": runner_version,
        "fixtures": fixtures,
    }
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_CURRENT_TEST_CONTEXT, "value"):
                delattr(_CURRENT_TEST_CONTEXT, "value")
        else:
            _CURRENT_TEST_CONTEXT.value = previous


_CURRENT_RETRY_CONTEXT = threading.local()


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
    return f"  \033[2;37mattempts={attempt}/{max_attempts}\033[0m"


TEST_TMP_ENV_VAR = "TENZIR_TMP_DIR"
_TMP_KEEP_ENV_VAR = "TENZIR_KEEP_TMP_DIRS"
_TMP_ROOT_NAME = ".tenzir-test"
_TMP_SUBDIR_NAME = "tmp"
_TMP_BASE_DIRS: set[Path] = set()
_ACTIVE_TMP_DIRS: set[Path] = set()
KEEP_TMP_DIRS = bool(os.environ.get(_TMP_KEEP_ENV_VAR))


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
    **kwargs: Any,
) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
    """Execute a subprocess honoring passthrough configuration.

    When passthrough is enabled the process inherits stdout/stderr so developers
    can observe output directly. Otherwise the helper captures both streams when
    `capture_output` is true, mirroring ``subprocess.run``'s behaviour.

    Runner authors should prefer this helper over direct ``subprocess`` calls so
    passthrough semantics remain consistent across implementations.
    """

    if any(key in kwargs for key in {"stdout", "stderr", "capture_output"}):
        raise TypeError("run_subprocess manages stdout/stderr automatically")

    passthrough = is_passthrough_enabled()
    stream_output = passthrough and not force_capture
    stdout = subprocess.PIPE if capture_output and not stream_output else None
    stderr = subprocess.PIPE if capture_output and not stream_output else None

    return subprocess.run(
        args,
        check=check,
        stdout=stdout,
        stderr=stderr,
        text=text,
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
    base = _resolve_tmp_base()
    prefix = f"{_tmp_prefix_for(test)}-"
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
    _ACTIVE_TMP_DIRS.discard(tmp_path)
    if KEEP_TMP_DIRS:
        return
    shutil.rmtree(tmp_path, ignore_errors=True)
    _cleanup_tmp_base_dirs()


def _cleanup_remaining_tmp_dirs() -> None:
    for tmp_path in list(_ACTIVE_TMP_DIRS):
        cleanup_test_tmp_dir(tmp_path)
    _cleanup_tmp_base_dirs()


def _cleanup_all_tmp_dirs() -> None:
    """Eagerly remove any temporary directories created by the harness."""

    if KEEP_TMP_DIRS:
        return
    for tmp_path in list(_ACTIVE_TMP_DIRS):
        cleanup_test_tmp_dir(tmp_path)
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

_verbose_logging = bool(
    os.environ.get("TENZIR_TEST_VERBOSE") or os.environ.get("TENZIR_TEST_LOG_COMPARISONS")
)
_debug_logging = bool(os.environ.get("TENZIR_TEST_DEBUG"))

_runner_names: set[str] = set()
_allowed_extensions: set[str] = set()
_DEFAULT_RUNNER_BY_SUFFIX: dict[str, str] = {
    ".tql": "tenzir",
    ".py": "python",
    ".sh": "shell",
}

_CONFIG_FILE_NAME = "test.yaml"
_CONFIG_LOGGER = logging.getLogger("tenzir_test.config")
_CONFIG_LOGGER.setLevel(logging.INFO)
if not _CONFIG_LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _CONFIG_LOGGER.addHandler(handler)
    _CONFIG_LOGGER.propagate = False

_DIRECTORY_CONFIG_CACHE: dict[Path, "_DirectoryConfig"] = {}

_DISCOVERY_ENABLED = False


def _set_discovery_logging(enabled: bool) -> None:
    global _DISCOVERY_ENABLED
    _DISCOVERY_ENABLED = enabled


def _print_discovery_message(message: str) -> None:
    print(f"{DEBUG_PREFIX} {message}")


class ProjectMarker(enum.Enum):
    """Sentinel indicators that describe a project root."""

    PACKAGE_MANIFEST = "package_manifest"
    TESTS_DIRECTORY = "tests_directory"
    TEST_CONFIG = "test_config"
    TEST_SUITE_DIRECTORY = "test_suite_directory"


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


def _format_project_heading(selection: ProjectSelection, *, base_root: Path) -> str:
    name = selection.root.name or selection.root.as_posix()
    relative = _format_relative_path(selection.root, base_root)
    return f"{name} ({relative})"


def _print_execution_plan(plan: ExecutionPlan, *, display_base: Path) -> None:
    active: list[tuple[str, ProjectSelection]] = []
    if plan.root.should_run():
        active.append(("■", plan.root))
    for satellite in plan.satellites:
        if satellite.should_run():
            active.append(("□", satellite))

    if not active:
        return

    if len(active) == 1:
        marker, selection = active[0]
        heading = _format_project_heading(selection, base_root=display_base)
        print(f"{INFO} executing project: {heading}")
        return

    print(f"{INFO} executing {len(active)} projects")
    for marker, selection in active:
        heading = _format_project_heading(selection, base_root=display_base)
        print(f"{INFO}   {marker} {heading}")


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
        "fixtures": tuple(),
        "inputs": None,
        "retry": 1,
    }


def _canonical_config_key(key: str) -> str:
    if key == "fixture":
        return "fixtures"
    return key


def _raise_config_error(location: Path | str, message: str, line_number: int | None = None) -> None:
    base = str(location)
    if line_number is not None:
        base = f"{base}:{line_number}"
    raise ValueError(f"Error in {base}: {message}")


def _normalize_fixtures_value(
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> tuple[str, ...]:
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
        return tuple()

    fixtures: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            _raise_config_error(
                location,
                f"Invalid fixture entry '{entry}', expected string",
                line_number,
            )
        name = entry.strip()
        if not name:
            _raise_config_error(
                location,
                "Fixture names must be non-empty strings",
                line_number,
            )
        fixtures.append(name)
    return tuple(fixtures)


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
    return None


def _assign_config_option(
    config: TestConfig,
    key: str,
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> None:
    canonical = _canonical_config_key(key)
    valid_keys: set[str] = {"error", "timeout", "runner", "skip", "fixtures", "inputs", "retry"}
    if canonical not in valid_keys:
        _raise_config_error(location, f"Unknown configuration key '{key}'", line_number)

    if canonical == "skip":
        if not isinstance(value, str) or not value.strip():
            _raise_config_error(
                location,
                "'skip' value must be a non-empty string",
                line_number,
            )
        config[canonical] = value
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
        return

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
            return
        if timeout_value <= 0:
            _raise_config_error(
                location,
                f"Invalid value for 'timeout', expected positive integer, got '{value}'",
                line_number,
            )
        config[canonical] = timeout_value
        return

    if canonical == "fixtures":
        config[canonical] = _normalize_fixtures_value(
            value, location=location, line_number=line_number
        )
        return

    if canonical == "inputs":
        config[canonical] = _normalize_inputs_value(
            value, location=location, line_number=line_number
        )
        return
    if canonical == "retry":
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
            return
        if retry_value <= 0:
            _raise_config_error(
                location,
                f"Invalid value for 'retry', expected positive integer, got '{value}'",
                line_number,
            )
        config[canonical] = retry_value
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
            _CONFIG_LOGGER.info(
                "Runner '%s' is not registered; proceeding with explicit selection.",
                value,
            )
        config[canonical] = value
        return

    config[canonical] = value


def _log_directory_override(
    *,
    path: Path,
    key: str,
    previous: object,
    new: object,
    previous_source: Path,
) -> None:
    message = (
        f"{path} overrides '{key}' from {previous!r} (defined in {previous_source}) to {new!r}"
    )
    _CONFIG_LOGGER.info(message)


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


def _clear_directory_config_cache() -> None:
    _DIRECTORY_CONFIG_CACHE.clear()


def _iter_project_test_directories(root: Path) -> Iterator[Path]:
    """Yield directories that contain tests for the current project."""

    if EXECUTION_MODE is ExecutionMode.PACKAGE:
        if root.name == "tests" and root.is_dir():
            yield root
            return
        package_root = _DETECTED_PACKAGE_ROOT
        if package_root is not None:
            tests_dir = package_root / "tests"
            if tests_dir.is_dir():
                yield tests_dir
                return
        if root.is_dir():
            yield root
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
    """Return True when the path lives under an inputs directory."""
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        parts = path.parts

    for index, part in enumerate(parts):
        if part != "inputs":
            continue
        if index == 0:
            return True
        if index > 0 and parts[index - 1] == "tests":
            return True
    return False


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


def _resolve_inputs_dir(root: Path) -> Path:
    direct = root / "inputs"
    if direct.exists():
        return direct
    tests_inputs = root / "tests" / "inputs"
    if tests_inputs.exists():
        return tests_inputs
    return direct


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
        inputs_path = str(_resolve_inputs_dir(ROOT).resolve())
    else:
        candidate = Path(os.fspath(inputs))
        if not candidate.is_absolute():
            candidate = test.parent / candidate
        inputs_path = str(candidate.resolve())
    env["TENZIR_INPUTS"] = inputs_path
    if config_file.exists():
        env.setdefault("TENZIR_CONFIG", str(config_file))
    if node_config_file.exists():
        env["TENZIR_NODE_CONFIG"] = str(node_config_file)
    if TENZIR_BINARY:
        env["TENZIR_BINARY"] = TENZIR_BINARY
    if TENZIR_NODE_BINARY:
        env["TENZIR_NODE_BINARY"] = TENZIR_NODE_BINARY
    env["TENZIR_TEST_ROOT"] = str(ROOT)
    tmp_dir = _create_test_tmp_dir(test)
    env[TEST_TMP_ENV_VAR] = str(tmp_dir)
    return env, config_args


def _apply_fixture_env(env: dict[str, str], fixtures: tuple[str, ...]) -> None:
    if fixtures:
        env["TENZIR_TEST_FIXTURES"] = ",".join(fixtures)
    else:
        env.pop("TENZIR_TEST_FIXTURES", None)


def enable_comparison_logging(enabled: bool) -> None:
    global _verbose_logging
    _verbose_logging = enabled


def log_comparison(test: Path, ref_path: Path, *, mode: str) -> None:
    if not _verbose_logging or should_suppress_failure_output():
        return
    rel_test = _relativize_path(test)
    rel_ref = _relativize_path(ref_path)
    print(f"{VERBOSE_PREFIX} {mode} {rel_test} -> {rel_ref}")


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
                _assign_config_option(config, str(key), value, location=test_file)

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
            _assign_config_option(
                config,
                key,
                value,
                location=test_file,
                line_number=line_number,
            )

    if coverage:
        timeout_value = cast(int, config["timeout"])
        config["timeout"] = timeout_value * 5

    runner_value = config.get("runner")
    if not isinstance(runner_value, str) or not runner_value:
        suffix = test_file.suffix.lower()
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
    failed_paths: list[Path] = dataclasses.field(default_factory=list)
    skipped_paths: list[Path] = dataclasses.field(default_factory=list)
    runner_stats: dict[str, RunnerStats] = dataclasses.field(default_factory=dict)
    fixture_stats: dict[str, FixtureStats] = dataclasses.field(default_factory=dict)

    def __add__(self, other: "Summary") -> "Summary":
        return Summary(
            failed=self.failed + other.failed,
            total=self.total + other.total,
            skipped=self.skipped + other.skipped,
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


def _format_percentage(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


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

    return f"Test summary: {passed_segment} • {failed_segment} • {skipped_segment}"


def _format_run_context_suffix() -> str:
    if not SHOW_TEST_DETAILS:
        return ""
    context = getattr(_CURRENT_TEST_CONTEXT, "value", None)
    if not isinstance(context, dict):
        return ""

    runner_name = context.get("runner_name")
    fixtures = context.get("fixtures") or tuple()
    parts: list[str] = []
    if isinstance(runner_name, str) and runner_name:
        parts.append(f"runner={runner_name}")
    if isinstance(fixtures, tuple) and fixtures:
        parts.append(f"fixtures={', '.join(fixtures)}")
    if not parts:
        return ""
    details = ", ".join(parts)
    return f"  \033[2;37m{details}\033[0m"


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


def _collect_runner_versions(
    queue: Sequence[RunnerQueueItem],
    *,
    tenzir_version: str | None,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    if tenzir_version:
        versions["tenzir"] = tenzir_version

    for runner, _ in queue:
        attr = getattr(runner, "version", None)
        if isinstance(attr, str) and attr:
            versions.setdefault(runner.name, attr)
    return versions


def _runner_breakdown(
    queue: Sequence[RunnerQueueItem],
    *,
    tenzir_version: str | None,
    runner_versions: Mapping[str, str] | None = None,
) -> list[tuple[str, int, str | None]]:
    counts: dict[str, int] = {}
    for runner, _ in queue:
        counts[runner.name] = counts.get(runner.name, 0) + 1

    breakdown: list[tuple[str, int, str | None]] = []
    for name in sorted(counts):
        version = (runner_versions or {}).get(name)
        if version is None and name == "tenzir":
            version = tenzir_version
        breakdown.append((name, counts[name], version))
    return breakdown


def _print_aggregate_totals(project_count: int, summary: Summary) -> None:
    total = summary.total
    failed = summary.failed
    skipped = summary.skipped
    passed = total - failed - skipped
    print(
        f"{INFO} aggregate totals across {project_count} projects: "
        f"{total} tests (passed={passed}, failed={failed}, skipped={skipped})"
    )


def _summarize_harness_configuration(
    *,
    jobs: int,
    update: bool,
    coverage: bool,
    verbose: bool,
    debug: bool,
    runner_summary: bool,
    fixture_summary: bool,
    passthrough: bool,
) -> tuple[int, str]:
    enabled_flags: list[str] = []
    toggles = (
        ("update", update),
        ("coverage", coverage),
        ("verbose", verbose),
        ("debug", debug),
        ("runner-summary", runner_summary),
        ("fixture-summary", fixture_summary),
        ("keep-tmp-dirs", KEEP_TMP_DIRS),
    )
    for name, flag in toggles:
        if flag:
            enabled_flags.append(name)
    if passthrough:
        enabled_flags.append("passthrough")
    return jobs, ", ".join(enabled_flags)


def _relativize_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        try:
            relative = os.path.relpath(path, ROOT)
        except ValueError:
            return path
        return Path(relative)


def _get_test_fixtures(test: Path, *, coverage: bool) -> tuple[str, ...]:
    try:
        config = parse_test_config(test, coverage=coverage)
    except ValueError:
        return tuple()
    fixtures = config.get("fixtures", tuple())
    if isinstance(fixtures, tuple):
        return typing.cast(tuple[str, ...], fixtures)
    return tuple(typing.cast(Iterable[str], fixtures))


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
        if char in glyphs:
            parts.append(f"{color}{char}\033[0m")
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
    rows = [
        (f"{CHECKMARK} Passed", str(passed), _format_percentage(passed, total)),
        (f"{SKIP} Skipped", str(summary.skipped), _format_percentage(summary.skipped, total)),
        (f"{CROSS} Failed", str(summary.failed), _format_percentage(summary.failed, total)),
        ("Total tests", str(total), "100.0%"),
    ]
    headers = ("Outcome", "Count", "Share")
    label_width = max(len(headers[0]), *(len(_strip_ansi(row[0])) for row in rows))
    count_width = max(len(headers[1]), *(len(row[1]) for row in rows))
    share_width = max(len(headers[2]), *(len(row[2]) for row in rows))

    def frame(char: str, width: int) -> str:
        return char * (width + 2)

    top = f"┌{frame('─', label_width)}┬{frame('─', count_width)}┬{frame('─', share_width)}┐"
    header = (
        f"│ {_ljust_visible(headers[0], label_width)} │ {headers[1].rjust(count_width)} │ "
        f"{headers[2].rjust(share_width)} │"
    )
    separator = f"├{frame('─', label_width)}┼{frame('─', count_width)}┼{frame('─', share_width)}┤"
    body = [
        f"│ {_ljust_visible(label, label_width)} │ {_rjust_visible(count, count_width)} │ "
        f"{_rjust_visible(share, share_width)} │"
        for label, count, share in rows
    ]
    bottom = f"└{frame('─', label_width)}┴{frame('─', count_width)}┴{frame('─', share_width)}┘"
    return [top, header, separator, *body, bottom]


def _print_ascii_summary(summary: Summary, *, include_runner: bool, include_fixture: bool) -> None:
    runner_lines = _render_runner_box(summary) if include_runner else []
    fixture_lines = _render_fixture_box(summary) if include_fixture else []
    outcome_lines = _render_summary_box(summary)
    if not runner_lines and not fixture_lines and not outcome_lines:
        return
    print()
    segments: list[list[str]] = []
    if runner_lines:
        segments.append(runner_lines)
    if fixture_lines:
        segments.append(fixture_lines)
    if outcome_lines:
        segments.append(outcome_lines)
    for index, block in enumerate(segments):
        for line in block:
            print(line)
        if index < len(segments) - 1:
            print()


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
            print(f"  {_color_tree_glyphs(line, '\033[31m')}")


def get_version() -> str:
    if not TENZIR_BINARY:
        raise FileNotFoundError("TENZIR_BINARY is not configured")
    return (
        subprocess.check_output(
            [
                TENZIR_BINARY,
                "--bare-mode",
                "--console-verbosity=warning",
                "version | select version | write_lines",
            ]
        )
        .decode()
        .strip()
    )


def success(test: Path) -> None:
    with stdout_lock:
        rel_test = _relativize_path(test)
        suffix = _format_run_context_suffix()
        attempt_suffix = _format_attempt_suffix()
        print(f"{CHECKMARK} {rel_test}{suffix}{attempt_suffix}")


def fail(test: Path) -> None:
    with stdout_lock:
        rel_test = _relativize_path(test)
        suffix = _format_run_context_suffix()
        attempt_suffix = _format_attempt_suffix()
        print(f"{CROSS} {rel_test}{suffix}{attempt_suffix}")


def last_and(items: Iterable[T]) -> Iterator[tuple[bool, T]]:
    iterator = iter(items)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for item in iterator:
        yield (False, previous)
        previous = item
    yield (True, previous)


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
    with stdout_lock:
        rel_path = _relativize_path(path)
        skip = 2
        for i, line in enumerate(diff):
            if skip > 0:
                skip -= 1
                continue
            if line.startswith(b"@@"):
                print(f"┌─▶ \033[31m{rel_path}\033[0m")
                continue
            if line.startswith(b"+"):
                line = b"\033[92m" + line + b"\033[0m"
            elif line.startswith(b"-"):
                line = b"\033[31m" + line + b"\033[0m"
            prefix = ("│ " if i != len(diff) - 1 else "└─").encode()
            sys.stdout.buffer.write(prefix + line)


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
        report_failure(test, f"└─▶ \033[31m{e}\033[0m")
        return False

    inputs_override = typing.cast(str | None, test_config.get("inputs"))
    env, config_args = get_test_env_and_config_args(test, inputs=inputs_override)
    fixtures = cast(tuple[str, ...], test_config.get("fixtures", tuple()))
    timeout = cast(int, test_config["timeout"])
    expect_error = bool(test_config.get("error", False))
    passthrough_mode = is_passthrough_enabled()

    context_token = fixtures_impl.push_context(
        fixtures_impl.FixtureContext(
            test=test,
            config=cast(dict[str, Any], test_config),
            coverage=coverage,
            env=env,
            config_args=tuple(config_args),
            tenzir_binary=TENZIR_BINARY,
            tenzir_node_binary=TENZIR_NODE_BINARY,
        )
    )
    try:
        with fixtures_impl.activate(fixtures) as fixture_env:
            env.update(fixture_env)
            _apply_fixture_env(env, fixtures)

            package_root = packages.find_package_root(test)
            package_args: list[str] = []
            if package_root is not None:
                env["TENZIR_PACKAGE_ROOT"] = str(package_root)
                package_tests_root = package_root / "tests"
                if inputs_override is None:
                    env["TENZIR_INPUTS"] = str(package_tests_root / "inputs")
                package_args.append(f"--package-dirs={package_root}")

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
            node_requested = "node" in fixtures
            if node_requested:
                endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                if not endpoint:
                    raise RuntimeError("node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT")
                node_args.append(f"--endpoint={endpoint}")

            if not TENZIR_BINARY:
                raise RuntimeError("TENZIR_BINARY must be configured before running tests")
            cmd: list[str] = [
                TENZIR_BINARY,
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
            completed = run_subprocess(
                cmd,
                timeout=timeout,
                env=env,
                capture_output=not passthrough_mode,
                cwd=str(ROOT),
            )
        good = completed.returncode == 0
        output = b""
        if not passthrough_mode:
            captured = completed.stdout or b""
            output = captured.replace(str(ROOT).encode() + b"/", b"")
    except subprocess.TimeoutExpired:
        report_failure(test, f"└─▶ \033[31msubprocess hit {timeout}s timeout\033[0m")
        return False
    except subprocess.CalledProcessError as e:
        report_failure(test, f"└─▶ \033[31msubprocess error: {e}\033[0m")
        return False
    except Exception as e:
        report_failure(test, f"└─▶ \033[31munexpected exception: {e}\033[0m")
        return False
    finally:
        fixtures_impl.pop_context(context_token)
        cleanup_test_tmp_dir(env.get(TEST_TMP_ENV_VAR))

    if expect_error == good:
        interrupted = _is_interrupt_exit(completed.returncode) or interrupt_requested()
        if should_suppress_failure_output() and not interrupted:
            return False
        if interrupted:
            _request_interrupt()
        message = (
            _INTERRUPTED_NOTICE
            if interrupted
            else f"┌─▶ \033[31mgot unexpected exit code {completed.returncode}\033[0m"
        )
        if passthrough_mode:
            report_failure(test, message)
        else:
            with stdout_lock:
                report_failure(test, message)
                if not interrupted:
                    for last, line in last_and(output.split(b"\n")):
                        prefix = "│ " if not last else "└─"
                        sys.stdout.buffer.write(prefix.encode() + line + b"\n")
        return False
    if passthrough_mode:
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
            report_failure(test, f'└─▶ \033[31mFailed to find ref file: "{ref_path}"\033[0m')
            return False
        log_comparison(test, ref_path, mode="comparing")
        expected = ref_path.read_bytes()
        if expected != output:
            if interrupt_requested():
                report_interrupted_test(test)
            else:
                report_failure(test, "")
                print_diff(expected, output, ref_path)
            return False
    success(test)
    return True


def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
    rel_path = _relativize_path(test)
    suffix = _format_run_context_suffix()
    print(f"{SKIP} skipped {rel_path}{suffix}: {reason}")
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
                    f'└─▶ \033[31mReference file for skipped test must be empty: "{ref_path}"\033[0m',
                )
                return False
    return "skipped"


def refresh_runner_metadata() -> None:
    _refresh_registry()


refresh_runner_metadata()


class Worker:
    def __init__(
        self,
        queue: list[RunnerQueueItem],
        *,
        update: bool,
        coverage: bool = False,
        runner_versions: Mapping[str, str] | None = None,
    ) -> None:
        self._queue = queue
        self._result: Summary | None = None
        self._exception: BaseException | None = None
        self._update = update
        self._coverage = coverage
        self._runner_versions = dict(runner_versions or {})
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

    def _work(self) -> Summary:
        try:
            self._result = Summary()
            result = self._result
            while True:
                if interrupt_requested():
                    break
                try:
                    item = self._queue.pop()
                except IndexError:
                    break

                runner, test_path = item
                if interrupt_requested():
                    break
                fixtures = _get_test_fixtures(test_path, coverage=self._coverage)
                retry_limit = 1
                retry_config: TestConfig | None = None
                try:
                    retry_config = parse_test_config(test_path, coverage=self._coverage)
                except ValueError:
                    retry_config = None
                if retry_config is not None:
                    raw_retry = retry_config.get("retry", 0)
                    if isinstance(raw_retry, int):
                        retry_limit = max(1, raw_retry)
                if is_passthrough_enabled():
                    rel_path = _relativize_path(test_path)
                    detail_bits = [f"runner={runner.name}"]
                    if fixtures:
                        detail_bits.append(f"fixtures={', '.join(fixtures)}")
                    detail_segment = f" ({', '.join(detail_bits)})" if detail_bits else ""
                    with stdout_lock:
                        print(f"{INFO} running {rel_path}{detail_segment} [passthrough]")
                max_attempts = retry_limit
                attempts = 0
                final_outcome: bool | str = False
                final_interrupted = False
                rel_path = _relativize_path(test_path)
                while attempts < max_attempts:
                    if interrupt_requested():
                        break
                    attempts += 1
                    with _push_retry_context(attempt=attempts, max_attempts=max_attempts):
                        context = (
                            _push_test_context(
                                runner_name=runner.name,
                                runner_version=self._runner_versions.get(runner.name),
                                fixtures=fixtures,
                            )
                            if SHOW_TEST_DETAILS
                            else contextlib.nullcontext()
                        )
                        interrupted = False
                        try:
                            with context:
                                outcome = runner.run(test_path, self._update, self._coverage)
                        except KeyboardInterrupt:  # pragma: no cover - defensive guard
                            _request_interrupt()
                            interrupted = True
                            outcome = False

                        if interrupted:
                            report_failure(test_path, _INTERRUPTED_NOTICE)
                            final_interrupted = True
                        final_outcome = outcome
                        if final_interrupted or interrupt_requested():
                            break
                        if outcome == "skipped" or outcome:
                            break
                        if attempts < max_attempts:
                            # keep the loop silent between attempts; only the final
                            # outcome is reported to avoid noisy logs
                            continue
                result.total += 1
                result.record_runner_outcome(runner.name, final_outcome)
                if fixtures:
                    result.record_fixture_outcome(fixtures, final_outcome)
                if final_outcome == "skipped":
                    result.skipped += 1
                    result.skipped_paths.append(rel_path)
                elif not final_outcome:
                    result.failed += 1
                    result.failed_paths.append(rel_path)
                if final_interrupted or interrupt_requested():
                    break
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            self._exception = exc
            if self._result is None:
                self._result = Summary()
            return self._result


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


def run_cli(
    *,
    root: Path | None,
    tenzir_binary: Path | None,
    tenzir_node_binary: Path | None,
    tests: Sequence[Path],
    update: bool,
    verbose: bool,
    debug: bool,
    purge: bool,
    coverage: bool,
    coverage_source_dir: Path | None,
    runner_summary: bool,
    fixture_summary: bool,
    jobs: int,
    keep_tmp_dirs: bool,
    show_test_details: bool,
    passthrough: bool,
    jobs_overridden: bool = False,
    all_projects: bool = False,
) -> None:
    from tenzir_test.engine import state as engine_state

    try:
        verbose_enabled = bool(verbose or _verbose_logging)
        debug_enabled = bool(debug or _debug_logging)

        fixture_logger = logging.getLogger("tenzir_test.fixtures")
        root_logger = logging.getLogger()

        _set_discovery_logging(debug_enabled)
        enable_comparison_logging(verbose_enabled or debug_enabled)

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
            fixture_logger.setLevel(logging.INFO)
            fixture_logger.propagate = True
        else:
            for existing_handler in list(root_logger.handlers):
                existing_handler.setFormatter(default_formatter)
            root_logger.setLevel(logging.WARNING)
            fixture_logger.setLevel(logging.WARNING)
            fixture_logger.propagate = True

        global SHOW_TEST_DETAILS
        set_keep_tmp_dirs(bool(os.environ.get(_TMP_KEEP_ENV_VAR)) or keep_tmp_dirs)
        SHOW_TEST_DETAILS = show_test_details
        if passthrough:
            harness_mode = HarnessMode.PASSTHROUGH
        elif update:
            harness_mode = HarnessMode.UPDATE
        else:
            harness_mode = HarnessMode.COMPARE
        set_harness_mode(harness_mode)
        passthrough_mode = harness_mode is HarnessMode.PASSTHROUGH
        if passthrough_mode and jobs > 1:
            if jobs_overridden:
                print(f"{INFO} forcing --jobs=1 in passthrough mode to preserve output ordering")
            jobs = 1
        if passthrough_mode and update:
            print(f"{INFO} ignoring --update in passthrough mode")
            update = False

        settings = discover_settings(
            root=root,
            tenzir_binary=tenzir_binary,
            tenzir_node_binary=tenzir_node_binary,
        )
        apply_settings(settings)
        selected_tests = list(tests)
        if not _is_project_root(ROOT):
            print(
                f"{INFO} no tenzir-test project detected at {ROOT}.\n"
                f"{INFO} run from your project root or provide --root."
            )
            if selected_tests:
                sample = ", ".join(str(path) for path in selected_tests[:3])
                if len(selected_tests) > 3:
                    sample += ", ..."
                print(f"{INFO} Ignoring provided selection(s): {sample}")
            sys.exit(1)

        plan = _build_execution_plan(
            ROOT,
            selected_tests,
            root_explicit=root is not None,
            all_projects=all_projects,
        )
        display_base = Path.cwd().resolve()
        _print_execution_plan(plan, display_base=display_base)

        with _install_interrupt_handler():
            engine_state.refresh()

            overall_summary = Summary()
            overall_queue_count = 0
            executed_projects: list[ProjectSelection] = []

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
                            sys.exit(f"error: {exc}")
                        refresh_runner_metadata()
                        _set_project_root(settings.root)
                        engine_state.refresh()
                    continue

                if executed_projects:
                    print()

                _set_project_root(selection.root)
                engine_state.refresh()
                expose_namespace = selection.kind == "root"
                try:
                    _load_project_runners(selection.root, expose_namespace=expose_namespace)
                    _load_project_fixtures(selection.root, expose_namespace=expose_namespace)
                except RuntimeError as exc:
                    sys.exit(f"error: {exc}")
                refresh_runner_metadata()

                tests_to_run = selection.selectors if not selection.run_all else [selection.root]

                if purge:
                    continue

                todo: set[RunnerQueueItem] = set()
                for test in tests_to_run:
                    if test.resolve() == selection.root.resolve():
                        all_tests = []
                        for tests_dir in _iter_project_test_directories(selection.root):
                            all_tests.extend(list(collect_all_tests(tests_dir)))
                        for test_path in all_tests:
                            try:
                                runner = get_runner_for_test(test_path)
                                todo.add((runner, test_path))
                            except ValueError as error:
                                sys.exit(f"error: {error}")
                        continue

                    resolved = test.resolve()
                    if not resolved.exists():
                        sys.exit(f"error: test path `{test}` does not exist")

                    if resolved.is_dir():
                        if _is_inputs_path(resolved):
                            continue
                        tql_files = list(collect_all_tests(resolved))
                        if not tql_files:
                            sys.exit(f"error: no {_allowed_extensions} files found in {resolved}")
                        for file_path in tql_files:
                            try:
                                runner = get_runner_for_test(file_path)
                                todo.add((runner, file_path))
                            except ValueError as error:
                                sys.exit(f"error: {error}")
                    elif resolved.is_file():
                        if _is_inputs_path(resolved):
                            continue
                        if resolved.suffix[1:] in _allowed_extensions:
                            try:
                                runner = get_runner_for_test(resolved)
                                todo.add((runner, resolved))
                            except ValueError as error:
                                sys.exit(f"error: {error}")
                        else:
                            sys.exit(
                                f"error: unsupported file type {resolved.suffix} for {resolved} - only {_allowed_extensions} files are supported"
                            )
                    else:
                        sys.exit(f"error: `{test}` is neither a file nor a directory")

                if interrupt_requested():
                    break

                queue = list(todo)
                queue.sort(key=lambda tup: str(tup[1]), reverse=True)
                project_queue_size = len(queue)
                job_count, enabled_flags = _summarize_harness_configuration(
                    jobs=jobs,
                    update=update,
                    coverage=coverage,
                    verbose=verbose_enabled,
                    debug=debug_enabled,
                    runner_summary=runner_summary,
                    fixture_summary=fixture_summary,
                    passthrough=passthrough_mode,
                )

                relative_path = _format_relative_path(selection.root, display_base)
                if not project_queue_size:
                    _print_project_start(
                        relative_path=relative_path,
                        queue_size=project_queue_size,
                        job_count=job_count,
                        enabled_flags=enabled_flags,
                    )
                    continue

                os.environ["TENZIR_EXEC__DUMP_DIAGNOSTICS"] = "true"
                if not TENZIR_BINARY:
                    sys.exit(f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`")
                try:
                    tenzir_version = get_version()
                except FileNotFoundError:
                    sys.exit(f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`")

                runner_versions = _collect_runner_versions(queue, tenzir_version=tenzir_version)
                runner_breakdown = _runner_breakdown(
                    queue,
                    tenzir_version=tenzir_version,
                    runner_versions=runner_versions,
                )

                _print_project_start(
                    relative_path=relative_path,
                    queue_size=project_queue_size,
                    job_count=job_count,
                    enabled_flags=enabled_flags,
                )
                count_width = max((len(str(count)) for _, count, _ in runner_breakdown), default=1)
                for name, count, version in runner_breakdown:
                    version_segment = f" (v{version})" if version else ""
                    print(f"{INFO}   {count:>{count_width}}× {name}{version_segment}")

                workers = [
                    Worker(
                        queue,
                        update=update,
                        coverage=coverage,
                        runner_versions=runner_versions,
                    )
                    for _ in range(jobs)
                ]
                project_summary = Summary()
                for worker in workers:
                    worker.start()
                try:
                    for worker in workers:
                        project_summary += worker.join()
                except KeyboardInterrupt:  # pragma: no cover - defensive guard
                    _request_interrupt()
                    for worker in workers:
                        worker.join()
                    break

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

                if interrupt_requested():
                    break

            # Restore root project context for subsequent operations.
            _set_project_root(settings.root)
            engine_state.refresh()

            if purge:
                for runner in runners_iter_runners():
                    runner.purge()
                return

            if overall_queue_count == 0:
                print(f"{INFO} no tests selected")
                return

            if len(executed_projects) > 1:
                _print_aggregate_totals(len(executed_projects), overall_summary)

            if interrupt_requested():
                sys.exit(130)

            if overall_summary.failed > 0:
                sys.exit(1)

    finally:
        _cleanup_all_tmp_dirs()


def main(argv: Sequence[str] | None = None) -> None:
    import click

    from . import cli as cli_module

    try:
        cli_module.cli.main(
            args=list(argv) if argv is not None else None,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Abort as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


def _print_project_start(
    *,
    relative_path: str,
    queue_size: int,
    job_count: int,
    enabled_flags: str,
) -> None:
    toggles = f"; {enabled_flags}" if enabled_flags else ""
    jobs_segment = f" ({job_count} jobs)" if job_count else ""
    print(f"{INFO} running {queue_size} tests{jobs_segment} in project {relative_path}{toggles}")
