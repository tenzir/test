from __future__ import annotations

import typing
from pathlib import Path

from ._utils import get_run_module
from .tql_runner import TqlRunner


class TenzirRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(name="tenzir")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        if test_config.get("skip"):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    str(test_config["skip"]),
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )
        return bool(
            run_mod.run_simple_test(
                test,
                update=update,
                output_ext=self.output_ext,
                coverage=coverage,
            )
        )


__all__ = ["TenzirRunner"]
