from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(slots=True)
class Settings:
    """Configuration values steering how the harness discovers binaries and data."""

    root: Path
    tenzir_binary: str | None
    tenzir_node_binary: str | None

    @property
    def inputs_dir(self) -> Path:
        direct = self.root / "inputs"
        if direct.exists():
            return direct
        nested = self.root / "tests" / "inputs"
        if nested.exists():
            return nested
        return direct


def _coerce_binary(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return str(value)


def discover_settings(
    *,
    root: Path | None = None,
    tenzir_binary: str | os.PathLike[str] | None = None,
    tenzir_node_binary: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Produce harness settings by combining CLI overrides with environment defaults."""

    environment = dict(env or os.environ)

    chosen_root = root or environment.get("TENZIR_TEST_ROOT") or Path.cwd()
    root_path = Path(chosen_root).resolve()

    binary_cli = _coerce_binary(tenzir_binary)
    binary_env = environment.get("TENZIR_BINARY")
    tenzir_path = binary_cli or binary_env or shutil.which("tenzir")

    node_cli = _coerce_binary(tenzir_node_binary)
    node_env = environment.get("TENZIR_NODE_BINARY")
    node_path = node_cli or node_env or shutil.which("tenzir-node")

    return Settings(root=root_path, tenzir_binary=tenzir_path, tenzir_node_binary=node_path)
