from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


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


def _resolve_binary(
    env_var: str | None,
    binary_name: str,
) -> tuple[str, ...] | None:
    """Resolve a binary with fallback to uvx."""
    if env_var:
        try:
            parts = tuple(shlex.split(env_var))
        except ValueError as e:
            raise ValueError(
                f"Invalid shell syntax in environment variable for {binary_name}: {e}"
            ) from e
        if not parts:
            raise ValueError(f"Empty command in environment variable for {binary_name}")
        return parts
    which_result = shutil.which(binary_name)
    if which_result:
        return (which_result,)
    if shutil.which("uvx"):
        if binary_name == "tenzir-node":
            return ("uvx", "--from", "tenzir", "tenzir-node")
        return ("uvx", binary_name)
    return None


def discover_settings(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Produce harness settings by combining CLI overrides with environment defaults."""

    environment = dict(os.environ if env is None else env)

    chosen_root = root or environment.get("TENZIR_TEST_ROOT") or Path.cwd()
    root_path = Path(chosen_root).resolve()

    tenzir_path = _resolve_binary(environment.get("TENZIR_BINARY"), "tenzir")
    node_path = _resolve_binary(environment.get("TENZIR_NODE_BINARY"), "tenzir-node")

    return Settings(root=root_path, tenzir_binary=tenzir_path, tenzir_node_binary=node_path)
