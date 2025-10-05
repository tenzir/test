"""Fixtures defined by the example satellite project."""

from __future__ import annotations

from typing import Iterator

from tenzir_test import fixture


@fixture(name="satellite_marker")
def satellite_marker_fixture() -> Iterator[dict[str, str]]:
    """Expose a marker so tests can assert fixture inheritance works."""

    yield {"SATELLITE_MARKER": "example-satellite"}
