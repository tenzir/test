from __future__ import annotations

import pytest

from tenzir_test import config


@pytest.fixture(autouse=True)
def _no_run_subcommand_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings discovery from running `tenzir run --help`.

    Without a `tenzir` on `PATH`, the probe would download Tenzir through `uvx`.
    """
    monkeypatch.setattr(config, "_supports_run_subcommand", lambda command: False)
