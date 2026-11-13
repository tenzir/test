from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# The subcommand that runs a pipeline locally. Tenzir v6.20 introduced it, and
# from v7 on, the unified command interface sends everything else to the
# platform CLI, so a bare `tenzir` no longer runs pipelines.
RUN_SUBCOMMAND = "run"

# The `tenzir` executable, also as a `uvx` tool with a version, such as
# `tenzir@7.0.0` or `tenzir==7.0.0`.
_TENZIR_EXECUTABLE = re.compile(r"tenzir(?:(?:@|==).+)?")

# A first `uvx tenzir` downloads the package, so the probe must outlast that.
_PROBE_TIMEOUT_SECONDS = 600


@dataclass(slots=True)
class Settings:
    """Configuration values steering how the harness discovers binaries and data."""

    root: Path
    tenzir_binary: tuple[str, ...] | None
    tenzir_node_binary: tuple[str, ...] | None

    @property
    def inputs_dir(self) -> Path:
        direct = self.root / "inputs"
        if direct.exists():
            return direct
        nested = self.root / "tests" / "inputs"
        if nested.exists():
            return nested
        return direct


@functools.cache
def _supports_run_subcommand(command: tuple[str, ...]) -> bool:
    """Return whether `command` accepts `run` as the subcommand for local pipelines.

    Releases before v6.20 have no `run` subcommand and take `run --help` for a
    malformed invocation, which fails.
    """
    try:
        completed = subprocess.run(
            [*command, RUN_SUBCOMMAND, "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _names_tenzir_executable(command: tuple[str, ...]) -> bool:
    """Return whether `command` ends with the `tenzir` executable itself."""
    return _TENZIR_EXECUTABLE.fullmatch(Path(command[-1]).name) is not None


def _with_run_subcommand(command: tuple[str, ...]) -> tuple[str, ...]:
    """Append `run` when `command` is the `tenzir` executable and supports it."""
    if _names_tenzir_executable(command) and _supports_run_subcommand(command):
        return (*command, RUN_SUBCOMMAND)
    return command


def _split_command(env_var: str, binary_name: str) -> tuple[str, ...]:
    try:
        parts = tuple(shlex.split(env_var))
    except ValueError as e:
        raise ValueError(
            f"Invalid shell syntax in environment variable for {binary_name}: {e}"
        ) from e
    if not parts:
        raise ValueError(f"Empty command in environment variable for {binary_name}")
    return parts


def _resolve_binary(
    env_var: str | None,
    binary_name: str,
) -> tuple[str, ...] | None:
    """Resolve a binary with fallback to uvx."""
    if env_var:
        return _split_command(env_var, binary_name)
    which_result = shutil.which(binary_name)
    if which_result:
        return (which_result,)
    if shutil.which("uvx"):
        if binary_name == "tenzir-node":
            return ("uvx", "--from", "tenzir", "tenzir-node")
        return ("uvx", binary_name)
    return None


def _resolve_tenzir_binary(env_var: str | None) -> tuple[str, ...] | None:
    """Resolve the command that runs a pipeline locally.

    When the command is the `tenzir` executable, as found on `PATH`, through
    `uvx`, or named in `TENZIR_BINARY`, this adds the `run` subcommand if the
    release has one. Commands that already end in `run`, or name another
    executable, stay as they are.
    """
    command = _resolve_binary(env_var, "tenzir")
    if command is None:
        return None
    return _with_run_subcommand(command)


def platform_command(tenzir_binary: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return the `tenzir` command without `run`, for platform commands such as
    `tenzir package fetch`, or None if `tenzir_binary` is no `tenzir` command.
    """
    if not tenzir_binary:
        return None
    command = tenzir_binary
    if command[-1] == RUN_SUBCOMMAND and len(command) > 1:
        command = command[:-1]
    if not _names_tenzir_executable(command):
        return None
    return command


def discover_settings(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Produce harness settings by combining CLI overrides with environment defaults."""

    environment = dict(os.environ if env is None else env)

    chosen_root = root or environment.get("TENZIR_TEST_ROOT") or Path.cwd()
    root_path = Path(chosen_root).resolve()

    tenzir_path = _resolve_tenzir_binary(environment.get("TENZIR_BINARY"))
    node_path = _resolve_binary(environment.get("TENZIR_NODE_BINARY"), "tenzir-node")

    return Settings(root=root_path, tenzir_binary=tenzir_path, tenzir_node_binary=node_path)
