#!/usr/bin/env python3


from __future__ import annotations

import atexit
import builtins
import dataclasses
import difflib
import enum
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import typing
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

import yaml

from . import fixtures as fixture_api
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
T = TypeVar("T")


class ExecutionMode(enum.Enum):
    """Supported discovery modes."""

    PROJECT = "project"
    PACKAGE = "package"


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
CHECKMARK = "\033[92;1m✔\033[0m"
CROSS = "\033[31m✘\033[0m"
INFO = "\033[94;1mi\033[0m"
SKIP = "\033[90;1m●\033[0m"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

stdout_lock = threading.RLock()

TEST_TMP_ENV_VAR = "TENZIR_TMP_DIR"
_TMP_KEEP_ENV_VAR = "TENZIR_KEEP_TMP_DIRS"
_TMP_ROOT_NAME = ".tenzir-test"
_TMP_SUBDIR_NAME = "tmp"
_ACTIVE_TMP_DIRS: set[Path] = set()
KEEP_TMP_DIRS = bool(os.environ.get(_TMP_KEEP_ENV_VAR))


def _resolve_tmp_base() -> Path:
    preferred = ROOT / _TMP_ROOT_NAME / _TMP_SUBDIR_NAME
    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / _TMP_ROOT_NAME.strip(".") / _TMP_SUBDIR_NAME
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return preferred


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


def _cleanup_remaining_tmp_dirs() -> None:
    for tmp_path in list(_ACTIVE_TMP_DIRS):
        cleanup_test_tmp_dir(tmp_path)


atexit.register(_cleanup_remaining_tmp_dirs)

_log_comparisons = bool(os.environ.get("TENZIR_TEST_LOG_COMPARISONS"))

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


def _assign_config_option(
    config: TestConfig,
    key: str,
    value: typing.Any,
    *,
    location: Path | str,
    line_number: int | None = None,
) -> None:
    canonical = _canonical_config_key(key)
    valid_keys: set[str] = {"error", "timeout", "runner", "skip", "fixtures"}
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
    global TENZIR_BINARY, TENZIR_NODE_BINARY, ROOT, INPUTS_DIR, EXECUTION_MODE
    global _DETECTED_PACKAGE_ROOT
    global _settings
    _settings = settings
    TENZIR_BINARY = settings.tenzir_binary
    TENZIR_NODE_BINARY = settings.tenzir_node_binary
    ROOT = settings.root
    INPUTS_DIR = _resolve_inputs_dir(settings.root)
    EXECUTION_MODE, _DETECTED_PACKAGE_ROOT = detect_execution_mode(ROOT)
    _clear_directory_config_cache()


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


_PROJECT_FIXTURES_IMPORTED = False
_PROJECT_RUNNERS_IMPORTED = False


def _load_project_fixtures(root: Path) -> None:
    global _PROJECT_FIXTURES_IMPORTED
    if _PROJECT_FIXTURES_IMPORTED:
        return

    fixtures_package = root / "fixtures"
    fixtures_module = root / "fixtures.py"

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
        elif fixtures_module.exists():
            alias_target = _import_module_from_path("_tenzir_project_fixtures", fixtures_module)
        if alias_target is not None and "fixtures" not in sys.modules:
            sys.modules["fixtures"] = alias_target
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"failed to load project fixtures: {exc}") from exc

    _PROJECT_FIXTURES_IMPORTED = True


def _load_project_runners(root: Path) -> None:
    global _PROJECT_RUNNERS_IMPORTED
    if _PROJECT_RUNNERS_IMPORTED:
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
        if alias_target is not None and "runners" not in sys.modules:
            sys.modules["runners"] = alias_target
    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"failed to load project runners: {exc}") from exc

    _PROJECT_RUNNERS_IMPORTED = True
    _refresh_registry()


def get_test_env_and_config_args(test: Path) -> tuple[dict[str, str], list[str]]:
    config_file = test.parent / "tenzir.yaml"
    config_args = [f"--config={config_file}"] if config_file.exists() else []
    env = os.environ.copy()
    inputs_path = str(_resolve_inputs_dir(ROOT))
    env["TENZIR_INPUTS"] = inputs_path
    if config_file.exists():
        env.setdefault("TENZIR_CONFIG", str(config_file))
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
    global _log_comparisons
    _log_comparisons = enabled


def log_comparison(test: Path, ref_path: Path, *, mode: str) -> None:
    if not _log_comparisons:
        return
    try:
        rel_test = test.relative_to(ROOT)
    except ValueError:
        rel_test = test
    try:
        rel_ref = ref_path.relative_to(ROOT)
    except ValueError:
        rel_ref = ref_path
    compare_glyph = "\033[95m⇄\033[0m"
    verbose_glyph = "\033[37m•\033[0m"
    print(f"{verbose_glyph} {mode} {rel_test} {compare_glyph} {rel_ref}")


def report_failure(test: Path, message: str) -> None:
    with stdout_lock:
        fail(test)
        if message:
            print(message)


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


def _relativize_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


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
        for part in path.parts:
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
        print(f"{CHECKMARK} {test.relative_to(ROOT)}")


def fail(test: Path) -> None:
    with stdout_lock:
        print(f"{CROSS} {test.relative_to(ROOT)}")


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
    diff = list(
        difflib.diff_bytes(
            difflib.unified_diff,
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            n=2,
        )
    )
    with stdout_lock:
        skip = 2
        for i, line in enumerate(diff):
            if skip > 0:
                skip -= 1
                continue
            if line.startswith(b"@@"):
                print(f"┌─▶ \033[31m{path.relative_to(ROOT)}\033[0m")
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

    env, config_args = get_test_env_and_config_args(test)
    fixtures = cast(tuple[str, ...], test_config.get("fixtures", tuple()))
    timeout = cast(int, test_config["timeout"])
    expect_error = bool(test_config.get("error", False))

    context_token = fixture_api.push_context(
        fixture_api.FixtureContext(
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
        with fixture_api.activate(fixtures) as fixture_env:
            env.update(fixture_env)
            _apply_fixture_env(env, fixtures)

            package_root = packages.find_package_root(test)
            package_args: list[str] = []
            if package_root is not None:
                env["TENZIR_PACKAGE_ROOT"] = str(package_root)
                package_tests_root = package_root / "tests"
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
            completed = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=subprocess.PIPE,
                env=env,
            )
        output = completed.stdout
        output = output.replace(str(ROOT).encode() + b"/", b"")
        good = completed.returncode == 0
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
        fixture_api.pop_context(context_token)
        cleanup_test_tmp_dir(env.get(TEST_TMP_ENV_VAR))

    if expect_error == good:
        with stdout_lock:
            report_failure(
                test,
                f"┌─▶ \033[31mgot unexpected exit code {completed.returncode}\033[0m",
            )
            for last, line in last_and(output.split(b"\n")):
                prefix = "│ " if not last else "└─"
                sys.stdout.buffer.write(prefix.encode() + line + b"\n")
        return False
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
            report_failure(test, "")
            print_diff(expected, output, ref_path)
            return False
    success(test)
    return True


def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
    rel_path = test.relative_to(ROOT)
    print(f"{SKIP} skipped {rel_path}: {reason}")
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
    ) -> None:
        self._queue = queue
        self._result: Summary | None = None
        self._exception: BaseException | None = None
        self._update = update
        self._coverage = coverage
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
                try:
                    item = self._queue.pop()
                except IndexError:
                    break

                runner, test_path = item
                fixtures = _get_test_fixtures(test_path, coverage=self._coverage)
                outcome = runner.run(test_path, self._update, self._coverage)
                result.total += 1
                result.record_runner_outcome(runner.name, outcome)
                if fixtures:
                    result.record_fixture_outcome(fixtures, outcome)
                rel_path = _relativize_path(test_path)
                if outcome == "skipped":
                    result.skipped += 1
                    result.skipped_paths.append(rel_path)
                elif not outcome:
                    result.failed += 1
                    result.failed_paths.append(rel_path)
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
    log_comparisons: bool,
    purge: bool,
    coverage: bool,
    coverage_source_dir: Path | None,
    runner_summary: bool,
    fixture_summary: bool,
    jobs: int,
    keep_tmp_dirs: bool,
) -> None:
    from tenzir_test.engine import state as engine_state

    verbosity_enabled = bool(log_comparisons or _log_comparisons)
    fixture_logger = logging.getLogger("tenzir_test.fixtures")
    if verbosity_enabled:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.getLogger().setLevel(logging.INFO)
        fixture_logger.setLevel(logging.INFO)
    else:
        fixture_logger.setLevel(logging.WARNING)

    set_keep_tmp_dirs(bool(os.environ.get(_TMP_KEEP_ENV_VAR)) or keep_tmp_dirs)

    settings = discover_settings(
        root=root,
        tenzir_binary=tenzir_binary,
        tenzir_node_binary=tenzir_node_binary,
    )
    apply_settings(settings)
    selected_tests = list(tests)
    if not selected_tests and not _looks_like_project_root(ROOT):
        print(
            f"{INFO} no tenzir-test project detected at {ROOT}.\n"
            f"{INFO} Run from your project root or provide --root."
        )
        sys.exit(1)
    _load_project_runners(settings.root)
    _load_project_fixtures(settings.root)
    refresh_runner_metadata()
    enable_comparison_logging(log_comparisons or _log_comparisons)
    engine_state.refresh()

    tests_to_run = selected_tests or [ROOT]
    if purge:
        for runner in runners_iter_runners():
            runner.purge()
        return

    # TODO Make sure that all tests are located in the `tests` directory.
    todo = set()
    for test in tests_to_run:
        if test.resolve() == ROOT:
            # Collect all tests in project-specific locations
            all_tests = []
            for tests_dir in _iter_project_test_directories(ROOT):
                all_tests.extend(list(collect_all_tests(tests_dir)))

            # Process each test file using its configuration
            for test_path in all_tests:
                try:
                    runner = get_runner_for_test(test_path)
                    todo.add((runner, test_path))
                except ValueError as e:
                    # Show the error and exit
                    sys.exit(f"error: {e}")
            continue

        resolved = test.resolve()
        if not resolved.exists():
            sys.exit(f"error: test path `{test}` does not exist")

        # If it's a directory, collect all tests in it
        if resolved.is_dir():
            if _is_inputs_path(resolved):
                continue
            # Look for TQL files and use their config
            tql_files = list(collect_all_tests(resolved))
            if not tql_files:
                sys.exit(f"error: no {_allowed_extensions} files found in {resolved}")

            for file_path in tql_files:
                try:
                    runner = get_runner_for_test(file_path)
                    todo.add((runner, file_path))
                except ValueError as e:
                    sys.exit(f"error: {e}")
        # If it's a file, determine the runner from its configuration
        elif resolved.is_file():
            if _is_inputs_path(resolved):
                continue
            if resolved.suffix[1:] in _allowed_extensions:
                try:
                    runner = get_runner_for_test(resolved)
                    todo.add((runner, resolved))
                except ValueError as e:
                    sys.exit(f"error: {e}")
            else:
                # Error for non-TQL files
                sys.exit(
                    f"error: unsupported file type {resolved.suffix} for {resolved} - only {_allowed_extensions} files are supported"
                )
        else:
            sys.exit(f"error: `{test}` is neither a file nor a directory")

    queue = list(todo)
    # Sort by test path (item[1])
    queue.sort(key=lambda tup: str(tup[1]), reverse=True)
    os.environ["TENZIR_EXEC__DUMP_DIAGNOSTICS"] = "true"
    if not TENZIR_BINARY:
        sys.exit(f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`")
    try:
        version = get_version()
    except FileNotFoundError:
        sys.exit(f"error: could not find TENZIR_BINARY executable `{TENZIR_BINARY}`")

    print(f"{INFO} running {len(queue)} tests with v{version}")

    # Pass coverage flag to workers
    workers = [Worker(queue, update=update, coverage=coverage) for _ in range(jobs)]
    summary = Summary()
    for worker in workers:
        worker.start()
    for worker in workers:
        summary += worker.join()
    _print_detailed_summary(summary)
    _print_ascii_summary(
        summary,
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
    if summary.failed > 0:
        sys.exit(1)


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
