from __future__ import annotations

from pathlib import Path

from ..config import Settings
from tenzir_test import run

TENZIR_BINARY: tuple[str, ...] | None = run.TENZIR_BINARY
ROOT: Path = run.ROOT
INPUTS_DIR: Path = run.INPUTS_DIR


def apply_settings(settings: Settings) -> None:
    run.apply_settings(settings)
    _refresh()


def _refresh() -> None:
    global TENZIR_BINARY, ROOT, INPUTS_DIR
    TENZIR_BINARY = run.TENZIR_BINARY
    ROOT = run.ROOT
    INPUTS_DIR = run.INPUTS_DIR


# Ensure module state reflects current run module state by default.
_refresh()


def refresh() -> None:
    _refresh()


__all__ = [
    "TENZIR_BINARY",
    "ROOT",
    "INPUTS_DIR",
    "apply_settings",
    "refresh",
]
