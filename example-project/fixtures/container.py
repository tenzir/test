"""Fixture that demonstrates FixtureUnavailable.

This fixture checks for a ``container-runtime-example`` binary that does not
exist, causing it to raise ``FixtureUnavailable`` on every run. Tests that
depend on this fixture and opt in via ``skip: {on: fixture-unavailable}`` are
gracefully skipped instead of failing.
"""

from __future__ import annotations

import shutil
from typing import Iterator

from tenzir_test.fixtures import FixtureUnavailable, fixture


@fixture()
def container() -> Iterator[dict[str, str]]:
    binary = shutil.which("container-runtime-example")
    if binary is None:
        raise FixtureUnavailable("container-runtime-example not found")
    yield {"CONTAINER_RUNTIME": binary}
