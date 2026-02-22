from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._utils import get_run_module
from .ext_runner import ExtRunner
from .runner import RequirementCheckResult


class TqlRunner(ExtRunner):
    output_ext: str = "txt"

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, ext="tql")

    def _is_operator_available(
        self,
        operator: str,
        *,
        env: Mapping[str, str],
        config_args: Sequence[str],
    ) -> bool:
        run_mod = get_run_module()
        binary = run_mod.TENZIR_BINARY
        if not binary:
            raise RuntimeError(
                "TENZIR_BINARY must be configured before evaluating suite requirements"
            )
        escaped = operator.replace("\\", "\\\\").replace('"', '\\"')
        probe = f'plugins | where name == "{escaped}"'
        completed = run_mod.run_subprocess(
            [*binary, *config_args, probe],
            env=dict(env),
            capture_output=True,
            check=False,
            force_capture=True,
            cwd=str(run_mod.ROOT),
        )
        stdout = completed.stdout or b""
        return bool(stdout.strip())

    def check_requirements(
        self,
        requirements: Mapping[str, Sequence[str]],
        *,
        env: Mapping[str, str],
        config_args: Sequence[str],
    ) -> RequirementCheckResult:
        unsupported: set[str] = set()
        missing: dict[str, tuple[str, ...]] = {}

        operators = requirements.get("operators")
        if operators:
            missing_ops = [
                operator
                for operator in operators
                if not self._is_operator_available(
                    operator,
                    env=env,
                    config_args=config_args,
                )
            ]
            if missing_ops:
                missing["operators"] = tuple(missing_ops)

        for key, values in requirements.items():
            if key == "operators":
                continue
            if values:
                unsupported.add(key)

        return RequirementCheckResult(
            unsupported_keys=tuple(sorted(unsupported)),
            missing_values=missing,
        )


__all__ = ["TqlRunner"]
