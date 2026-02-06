from __future__ import annotations

import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from tenzir_test.run import SkipConfig

from ._utils import get_run_module
from .tql_runner import TqlRunner


class TenzirRunner(TqlRunner):
    def __init__(self) -> None:
        super().__init__(name="tenzir")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        skip_cfg: SkipConfig | None = test_config.get("skip")
        if skip_cfg is not None:
            if skip_cfg.is_static:
                assert skip_cfg.reason is not None
                return typing.cast(
                    bool | str,
                    run_mod.handle_skip(
                        skip_cfg.reason,
                        test,
                        update=update,
                        output_ext=self.output_ext,
                    ),
                )
            if skip_cfg.is_conditional:
                raise ValueError(
                    f"Conditional 'skip' config (on: fixture-unavailable) "
                    f"is a suite-level directive and cannot be handled by "
                    f"individual test runners. Test: {test}"
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
