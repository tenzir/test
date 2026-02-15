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
from tenzir_test.fixtures.container_runtime import detect_runtime


@fixture()
def container() -> Iterator[dict[str, str]]:
    runtime = detect_runtime(order=("container-runtime-example",))
    if runtime is None:
        raise FixtureUnavailable("container-runtime-example not found")
    runtime_path = shutil.which(runtime.binary)
    assert runtime_path is not None
    yield {"CONTAINER_RUNTIME": runtime_path}
