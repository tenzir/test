"""Project-scoped fixtures for the example tenzir-test project."""

from __future__ import annotations

# Import submodules so their @fixture decorators run at import time.
from . import container, http, server  # noqa: F401

__all__ = ["container", "http", "server"]
