"""Test execution utilities for the Tenzir ecosystem."""

from importlib.metadata import PackageNotFoundError, version

from . import run
from .run import ExecutionResult, HarnessError, ProjectResult, ensure_settings, execute
from .config import Settings, discover_settings
from .fixtures import (
    Executor,
    FixtureHandle,
    FixtureSelection,
    FixtureController,
    activate,
    acquire_fixture,
    fixture,
    has,
    register,
    fixtures_api,
    require,
)

fixtures = fixtures_api

__all__ = [
    "__version__",
    "Executor",
    "FixtureHandle",
    "FixtureSelection",
    "FixtureController",
    "activate",
    "acquire_fixture",
    "fixture",
    "fixtures",
    "Settings",
    "discover_settings",
    "ensure_settings",
    "execute",
    "has",
    "register",
    "require",
    "ExecutionResult",
    "ProjectResult",
    "HarnessError",
    "run",
]


def _get_version() -> str:
    """Return the installed package version or a development placeholder."""
    try:
        return version("tenzir-test")
    except PackageNotFoundError:  # pragma: no cover - missing metadata during dev
        return "0.0.0"


__version__ = _get_version()
