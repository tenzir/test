"""Built-in fixture that manages a transient ``tenzir-node`` instance."""

from __future__ import annotations

import atexit
import functools
import logging
import os
import shlex
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING
import weakref

from . import (
    current_context,
    register,
    register_tmp_dir_cleanup,
    unregister_tmp_dir_cleanup,
)

if TYPE_CHECKING:
    from . import FixtureContext


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the spawned node process and ensure its group is gone."""

    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pgid = None

    try:
        process.terminate()
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        if pgid is not None:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            # Leave a helpful error so call sites can report the leak.
            raise RuntimeError("tenzir-node left descendant processes running")


@dataclass(slots=True)
class _TempDirRecord:
    temp_dir: tempfile.TemporaryDirectory[str]
    ref: weakref.ref["FixtureContext"]
    path: Path
    process: subprocess.Popen[str] | None = None


_TEMP_DIRS: dict[int, _TempDirRecord] = {}
_TEMP_DIR_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def _stop_process(record: _TempDirRecord) -> None:
    process = record.process
    if process is None:
        return
    record.process = None
    try:
        _terminate_process(process)
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("failed to terminate tenzir-node process: %s", exc)


def _cleanup_record(record: _TempDirRecord) -> None:
    _stop_process(record)
    try:
        record.temp_dir.cleanup()
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.debug("failed to remove temporary directory %s: %s", record.path, exc)


def _cleanup_record_by_key(key: int) -> None:
    with _TEMP_DIR_LOCK:
        record = _TEMP_DIRS.pop(key, None)
    if record is None:
        return
    unregister_tmp_dir_cleanup(record.path)
    _cleanup_record(record)


def _ensure_temp_dir(context: "FixtureContext") -> Path:
    key = id(context)
    stale_record: _TempDirRecord | None = None
    with _TEMP_DIR_LOCK:
        record = _TEMP_DIRS.get(key)
        if record is not None:
            current = record.ref()
            if current is context:
                return record.path
            _TEMP_DIRS.pop(key, None)
            stale_record = record
    if stale_record is not None:
        unregister_tmp_dir_cleanup(stale_record.path)
        _cleanup_record(stale_record)

    tmp_root = context.env.get("TENZIR_TMP_DIR") or None
    temp_dir = tempfile.TemporaryDirectory(prefix="tenzir-node-", dir=tmp_root)
    path = Path(temp_dir.name)

    def _cleanup(dead_ref: weakref.ref["FixtureContext"]) -> None:
        with _TEMP_DIR_LOCK:
            existing = _TEMP_DIRS.get(key)
        if existing is not None and existing.ref is dead_ref:
            _cleanup_record_by_key(key)

    ctx_ref = weakref.ref(context, _cleanup)
    record = _TempDirRecord(temp_dir=temp_dir, ref=ctx_ref, path=path)
    with _TEMP_DIR_LOCK:
        _TEMP_DIRS[key] = record
    register_tmp_dir_cleanup(path, functools.partial(_cleanup_record_by_key, key))
    return path


@contextmanager
def node() -> Iterator[dict[str, str]]:
    """Start ``tenzir-node`` and yield environment data for dependent tests."""

    context = current_context()
    if context is None:
        raise RuntimeError("node fixture requires an active test context")

    node_binary = context.tenzir_node_binary or context.env.get("TENZIR_NODE_BINARY")
    if not node_binary:
        raise RuntimeError("TENZIR_NODE_BINARY must be configured for the node fixture")

    env = context.env.copy()
    config_args = [arg for arg in context.config_args if not arg.startswith("--config=")]
    node_config = env.get("TENZIR_NODE_CONFIG")
    if node_config:
        config_args.append(f"--config={node_config}")
    package_root = env.get("TENZIR_PACKAGE_ROOT")
    package_args: list[str] = []
    if package_root:
        package_arg = f"--package-dirs={package_root}"
        if package_arg not in config_args:
            package_args.append(package_arg)
    temp_dir = _ensure_temp_dir(context)
    key = id(context)

    configured_state = env.get("TENZIR_NODE_STATE_DIRECTORY")
    state_dir = Path(configured_state) if configured_state else temp_dir / "state"
    configured_cache = env.get("TENZIR_NODE_CACHE_DIRECTORY")
    cache_dir = Path(configured_cache) if configured_cache else temp_dir / "cache"
    state_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TENZIR_NODE_STATE_DIRECTORY", str(state_dir))
    env.setdefault("TENZIR_NODE_CACHE_DIRECTORY", str(cache_dir))

    if context.coverage:
        coverage_dir = env.get(
            "CMAKE_COVERAGE_OUTPUT_DIRECTORY",
            os.path.join(os.getcwd(), "coverage"),
        )
        source_dir = env.get("COVERAGE_SOURCE_DIR", os.getcwd())
        os.makedirs(coverage_dir, exist_ok=True)
        profile_path = os.path.join(
            coverage_dir,
            f"{context.test.stem}-node-%p.profraw",
        )
        env["LLVM_PROFILE_FILE"] = profile_path
        env["COVERAGE_SOURCE_DIR"] = source_dir

    test_root = context.test.parent

    node_cmd = [
        node_binary,
        "--bare-mode",
        "--console-verbosity=warning",
        f"--state-directory={state_dir}",
        f"--cache-directory={cache_dir}",
        "--endpoint=localhost:0",
        "--print-endpoint",
        *config_args,
        *package_args,
    ]

    if _LOGGER.isEnabledFor(logging.DEBUG):
        command_line = shlex.join(node_cmd)
        _LOGGER.debug("spawning tenzir-node: %s (cwd=%s)", command_line, test_root)

    process = subprocess.Popen(
        node_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
        cwd=test_root,
    )
    with _TEMP_DIR_LOCK:
        record = _TEMP_DIRS.get(key)
        if record is not None:
            record.process = process

    endpoint: str | None = None
    try:
        if process.stdout:
            endpoint = process.stdout.readline().strip()

        if not endpoint:
            raise RuntimeError("failed to obtain endpoint from tenzir-node")

        fixture_env = {
            "TENZIR_NODE_CLIENT_ENDPOINT": endpoint,
            "TENZIR_NODE_CLIENT_BINARY": context.tenzir_binary or env.get("TENZIR_BINARY"),
            "TENZIR_NODE_CLIENT_TIMEOUT": str(context.config["timeout"]),
            "TENZIR_NODE_STATE_DIRECTORY": str(state_dir),
            "TENZIR_NODE_CACHE_DIRECTORY": str(cache_dir),
        }
        # Filter out empty values to avoid polluting the environment.
        filtered_env = {k: v for k, v in fixture_env.items() if v}
        yield filtered_env
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        with _TEMP_DIR_LOCK:
            record = _TEMP_DIRS.get(key)
        if record is not None:
            _stop_process(record)
        else:  # pragma: no cover - defensive logging
            _LOGGER.debug("node fixture context missing record for key %s", key)
            try:
                _terminate_process(process)
            except Exception as exc:
                _LOGGER.warning("failed to terminate leaked tenzir-node (key=%s): %s", key, exc)


register("node", node, replace=True)


@atexit.register
def _cleanup_all_node_temp_dirs() -> None:  # pragma: no cover - interpreter shutdown
    keys: list[int]
    with _TEMP_DIR_LOCK:
        keys = list(_TEMP_DIRS.keys())
    for key in keys:
        _cleanup_record_by_key(key)


__all__ = ["node"]
