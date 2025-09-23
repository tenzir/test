#!/usr/bin/env python3

from __future__ import annotations

import argparse
import builtins
import dataclasses
import difflib
import importlib.util
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import typing
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Generator, TypeVar, cast

import yaml

from . import fixtures as fixture_api
from .config import Settings, discover_settings
from .runners import (
    CustomFixture,  # noqa: F401
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

_settings = discover_settings()
TENZIR_BINARY = _settings.tenzir_binary
TENZIR_NODE_BINARY = _settings.tenzir_node_binary
ROOT = _settings.root
INPUTS_DIR = _settings.inputs_dir
CHECKMARK = "\033[92;1m✓\033[0m"
CROSS = "\033[31m✘\033[0m"
INFO = "\033[94;1mi\033[0m"

stdout_lock = threading.RLock()

_log_comparisons = bool(os.environ.get("TENZIR_TEST_LOG_COMPARISONS"))

_runner_names: set[str] = set()
_allowed_extensions: set[str] = set()
_DEFAULT_RUNNER_BY_SUFFIX: dict[str, str] = {
    ".tql": "tenzir",
    ".py": "python",
}


def _is_inputs_path(path: Path) -> bool:
    """Return True when the path lives under the project root `inputs/` directory."""
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] == "inputs"


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
    return direct


def apply_settings(settings: Settings) -> None:
    global TENZIR_BINARY, TENZIR_NODE_BINARY, ROOT, INPUTS_DIR
    global _settings
    _settings = settings
    TENZIR_BINARY = settings.tenzir_binary
    TENZIR_NODE_BINARY = settings.tenzir_node_binary
    ROOT = settings.root
    INPUTS_DIR = _resolve_inputs_dir(settings.root)


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
    if TENZIR_BINARY:
        env["TENZIR_BINARY"] = TENZIR_BINARY
    if TENZIR_NODE_BINARY:
        env["TENZIR_NODE_BINARY"] = TENZIR_NODE_BINARY
    env["TENZIR_TEST_ROOT"] = str(ROOT)
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
    config: TestConfig = {
        "error": False,
        "timeout": 30,
        "runner": None,
        "skip": None,
        "fixtures": tuple(),
    }

    valid_keys: set[str] = set(config.keys()) | {"fixture"}
    is_tql = test_file.suffix == ".tql"
    is_py = test_file.suffix == ".py"

    def _raise(message: str, line_number: int | None = None) -> None:
        location = f"{test_file}:{line_number}" if line_number is not None else f"{test_file}"
        raise ValueError(f"Error in {location}: {message}")

    def _normalize_fixtures(value: typing.Any, line_number: int | None = None) -> tuple[str, ...]:
        items: list[str] = []
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
            _raise(
                f"Invalid value for 'fixtures', expected string or list, got '{value}'",
                line_number,
            )
            return tuple()

        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    name = entry.strip()
                    if not name:
                        _raise("Fixture names must be non-empty strings", line_number)
                    items.append(name)
                else:
                    _raise(
                        f"Invalid fixture entry '{entry}', expected string",
                        line_number,
                    )
        else:
            _raise(
                f"Invalid value for 'fixtures', expected list, got '{raw}'",
                line_number,
            )
        return tuple(items)

    def _assign(key: str, value: typing.Any, line_number: int | None = None) -> None:
        if key not in valid_keys:
            _raise(f"Unknown configuration key '{key}'", line_number)
        if key == "skip":
            if not isinstance(value, str) or not value.strip():
                _raise("'skip' value must be a non-empty string", line_number)
            config[key] = value
            return
        if key == "error":
            if isinstance(value, bool):
                config[key] = value
                return
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"true", "false"}:
                    config[key] = lowered == "true"
                    return
            _raise(
                f"Invalid value for '{key}', expected 'true' or 'false', got '{value}'",
                line_number,
            )
            return
        if key == "timeout":
            if isinstance(value, int):
                timeout_value = value
            elif isinstance(value, str) and value.strip().isdigit():
                timeout_value = int(value)
            else:
                _raise(
                    f"Invalid value for 'timeout', expected integer, got '{value}'",
                    line_number,
                )
                return
            if timeout_value <= 0:
                _raise(
                    f"Invalid value for 'timeout', expected positive integer, got '{value}'",
                    line_number,
                )
            config[key] = timeout_value
            return
        if key in {"fixture", "fixtures"}:
            config["fixtures"] = _normalize_fixtures(value, line_number)
            return
        if key == "runner":
            if not isinstance(value, str):
                _raise(
                    f"Invalid value for 'runner', expected string, got '{value}'",
                    line_number,
                )
            names = _runner_names or {runner.name for runner in runners_iter_runners()}
            if value not in names:
                _raise(
                    f"Invalid value for 'runner', expected one of {names}, got '{value}'",
                    line_number,
                )
            config[key] = value
            return
        config[key] = value

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
                _raise("YAML frontmatter must be terminated with '---'")
            yaml_data = yaml.safe_load("".join(yaml_lines)) or {}
            if not isinstance(yaml_data, dict):
                _raise("YAML frontmatter must define a mapping")
            for key, value in yaml_data.items():
                _assign(str(key), value)

    if not consumed_frontmatter:
        line_number = 0
        for raw_line in lines:
            line_number += 1
            stripped = raw_line.strip()
            if is_py and line_number == 1 and stripped.startswith("#!"):
                # Skip shebangs so subsequent frontmatter comments still apply.
                continue
            if is_tql:
                if not stripped.startswith("//"):
                    break
                content = stripped[2:].strip()
            elif is_py:
                if not stripped.startswith("#"):
                    break
                content = stripped[1:].strip()
            else:
                break
            parts = content.split(":", 1)
            if len(parts) != 2:
                if line_number == 1:
                    break
                _raise("Invalid frontmatter, expected 'key: value'", line_number)
            key = parts[0].strip()
            value = parts[1].strip()
            _assign(key, value, line_number)

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
            default_runner = matching_names[0] if matching_names else "tenzir"
        config["runner"] = default_runner
    return config


def print(*args: object, **kwargs: Any) -> None:
    # TODO: Properly solve the synchronization below.
    if "flush" not in kwargs:
        kwargs["flush"] = True
    builtins.print(*args, **kwargs)


@dataclasses.dataclass
class Summary:
    def __init__(self, failed: int = 0, total: int = 0, skipped: int = 0):
        self.failed = failed
        self.total = total
        self.skipped = skipped

    def __add__(self, other: "Summary") -> "Summary":
        return Summary(
            self.failed + other.failed,
            self.total + other.total,
            self.skipped + other.skipped,
        )


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


@contextmanager
def check_server(host: str = "127.0.0.1") -> Generator[int, None, None]:
    stop = False
    port_holder: dict[str, int | None] = {"port": None}

    def _serve() -> None:
        nonlocal stop
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, 0))
            sock.listen()
            port_holder["port"] = sock.getsockname()[1]
            while not stop:
                sock.settimeout(0.5)
                try:
                    conn, _addr = sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(0.5)
                    while not stop:
                        try:
                            data = conn.recv(1)
                        except socket.timeout:
                            continue
                        if not data:
                            break

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    while port_holder["port"] is None:
        time.sleep(0.05)
    try:
        port = port_holder["port"]
        assert port is not None
        yield port
    finally:
        stop = True
        thread.join()


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
    print(f"{INFO} skipped {rel_path}: {reason}")
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
                outcome = runner.run(test_path, self._update, self._coverage)
                result.total += 1
                if outcome == "skipped":
                    result.skipped += 1
                elif not outcome:
                    result.failed += 1
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


def main(argv: Sequence[str] | None = None) -> None:
    from tenzir_test.engine import state as engine_state

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--tenzir-binary", type=Path)
    parser.add_argument("--tenzir-node-binary", type=Path)
    parser.add_argument("tests", nargs="*", type=Path)
    parser.add_argument("-u", "--update", action="store_true")
    parser.add_argument(
        "-v",
        "--log-comparisons",
        action="store_true",
        help="Log reference comparison activity",
    )
    parser.add_argument("--purge", action="store_true")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable code coverage collection (increases timeouts by 5x)",
    )
    parser.add_argument(
        "--coverage-source-dir",
        type=str,
        help="Source directory for code coverage path mapping (defaults to current directory)",
    )
    default_jobs = 4 * (os.cpu_count() or 16)
    parser.add_argument("-j", "--jobs", type=int, default=default_jobs, metavar="N")
    args = parser.parse_args(argv)

    verbosity_enabled = bool(args.log_comparisons or _log_comparisons)
    fixture_logger = logging.getLogger("tenzir_test.fixtures")
    if verbosity_enabled:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.getLogger().setLevel(logging.INFO)
        fixture_logger.setLevel(logging.INFO)
    else:
        fixture_logger.setLevel(logging.WARNING)

    settings = discover_settings(
        root=args.root,
        tenzir_binary=args.tenzir_binary,
        tenzir_node_binary=args.tenzir_node_binary,
    )
    apply_settings(settings)
    _load_project_runners(settings.root)
    _load_project_fixtures(settings.root)
    refresh_runner_metadata()
    enable_comparison_logging(args.log_comparisons or _log_comparisons)
    engine_state.refresh()

    tests = list(args.tests) if args.tests else [ROOT]
    if args.purge:
        for runner in runners_iter_runners():
            runner.purge()
        return

    # TODO Make sure that all tests are located in the `tests` directory.
    todo = set()
    for test in tests:
        if test.resolve() == ROOT:
            # Collect all tests in all directories
            all_tests = []
            for dir_path in ROOT.iterdir():
                if dir_path.is_dir() and not dir_path.name.startswith("."):
                    if dir_path.name in {"fixtures", "runners"}:
                        continue
                    if _is_inputs_path(dir_path):
                        continue
                    all_tests.extend(list(collect_all_tests(dir_path)))

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
    workers = [Worker(queue, update=args.update, coverage=args.coverage) for _ in range(args.jobs)]
    summary = Summary()
    for worker in workers:
        worker.start()
    for worker in workers:
        summary += worker.join()
    print(
        f"{INFO} {summary.total - summary.failed - summary.skipped}/{summary.total} tests passed ({summary.skipped} skipped)"
    )
    if args.coverage:
        coverage_dir = os.environ.get(
            "CMAKE_COVERAGE_OUTPUT_DIRECTORY", os.path.join(os.getcwd(), "coverage")
        )
        source_dir = args.coverage_source_dir or os.getcwd()
        print(f"{INFO} Code coverage data collected in {coverage_dir}")
        print(f"{INFO} Source directory for coverage mapping: {source_dir}")
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
