"""Project-scoped fixtures for the example tenzir-test project."""

# Import modules with fixture registrations so decorators execute at import time.
from . import http

__all__ = ["http"]
