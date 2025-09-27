"""Test execution utilities for the Tenzir ecosystem."""

from . import run
from .config import Settings, discover_settings
from .fixtures import (
    Executor,
    FixtureHandle,
    FixtureSelection,
    activate,
    has,
    register,
    requested,
    require,
    startup,
    teardown,
)

__all__ = [
    "__version__",
    "Executor",
    "FixtureHandle",
    "FixtureSelection",
    "activate",
    "Settings",
    "discover_settings",
    "has",
    "register",
    "requested",
    "require",
    "startup",
    "teardown",
    "run",
]

from importlib.metadata import PackageNotFoundError, version


def _get_version() -> str:
    """Return the installed package version or a development placeholder."""
    try:
        return version("tenzir-test")
    except PackageNotFoundError:  # pragma: no cover - missing metadata during dev
        return "0.0.0"


__version__ = _get_version()
