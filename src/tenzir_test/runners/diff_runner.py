from __future__ import annotations

import difflib
import os
import typing
from pathlib import Path

from tenzir_test import fixtures as fixture_api

from ._utils import get_run_module
from .tql_runner import TqlRunner


class DiffRunner(TqlRunner):
    def __init__(self, *, a: str, b: str, name: str) -> None:
        super().__init__(name=name)
        self._a = a
        self._b = b

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool | str:
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test, coverage=coverage)
        skip_value = test_config.get("skip")
        if isinstance(skip_value, str):
            return typing.cast(
                bool | str,
                run_mod.handle_skip(
                    skip_value,
                    test,
                    update=update,
                    output_ext=self.output_ext,
                ),
            )

        inputs_override = typing.cast(str | None, test_config.get("inputs"))
        env, config_args = run_mod.get_test_env_and_config_args(test, inputs=inputs_override)
        fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
        node_requested = "node" in fixtures
        timeout = typing.cast(int, test_config["timeout"])

        context_token = fixture_api.push_context(
            fixture_api.FixtureContext(
                test=test,
                config=typing.cast(dict[str, typing.Any], test_config),
                coverage=coverage,
                env=env,
                config_args=tuple(config_args),
                tenzir_binary=run_mod.TENZIR_BINARY,
                tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
            )
        )
        try:
            with fixture_api.activate(fixtures) as fixture_env:
                env.update(fixture_env)
                run_mod._apply_fixture_env(env, fixtures)

                node_args: list[str] = []
                if node_requested:
                    endpoint = env.get("TENZIR_NODE_CLIENT_ENDPOINT")
                    if not endpoint:
                        raise RuntimeError(
                            "node fixture did not provide TENZIR_NODE_CLIENT_ENDPOINT"
                        )
                    node_args.append(f"--endpoint={endpoint}")

                binary = run_mod.TENZIR_BINARY
                if not binary:
                    raise RuntimeError("TENZIR_BINARY must be configured for diff runners")
                base_cmd: list[str] = [binary, *config_args]

                if coverage:
                    coverage_dir = env.get(
                        "CMAKE_COVERAGE_OUTPUT_DIRECTORY",
                        os.path.join(os.getcwd(), "coverage"),
                    )
                    source_dir = env.get("COVERAGE_SOURCE_DIR", os.getcwd())
                    os.makedirs(coverage_dir, exist_ok=True)
                    env["COVERAGE_SOURCE_DIR"] = source_dir
                    env["LLVM_PROFILE_FILE"] = os.path.join(
                        coverage_dir, f"{test.stem}-unopt-%p.profraw"
                    )

                unoptimized = run_mod.run_subprocess(
                    [*base_cmd, self._a, *node_args, "-f", str(test)],
                    timeout=timeout,
                    env=env,
                    capture_output=True,
                    check=False,
                    force_capture=True,
                )

                if coverage:
                    env["LLVM_PROFILE_FILE"] = os.path.join(
                        coverage_dir, f"{test.stem}-opt-%p.profraw"
                    )

                optimized = run_mod.run_subprocess(
                    [*base_cmd, self._b, *node_args, "-f", str(test)],
                    timeout=timeout,
                    env=env,
                    capture_output=True,
                    check=False,
                    force_capture=True,
                )
        finally:
            fixture_api.pop_context(context_token)
            run_mod.cleanup_test_tmp_dir(env.get(run_mod.TEST_TMP_ENV_VAR))

        # Strip the ROOT prefix from paths in output to make them relative,
        # consistent with run_simple_test behavior.
        root_bytes = str(run_mod.ROOT).encode() + b"/"
        unoptimized_stdout = unoptimized.stdout.replace(root_bytes, b"")
        optimized_stdout = optimized.stdout.replace(root_bytes, b"")
        diff_chunks = list(
            difflib.diff_bytes(
                difflib.unified_diff,
                unoptimized_stdout.splitlines(keepends=True),
                optimized_stdout.splitlines(keepends=True),
                n=2**31 - 1,
            )
        )[3:]
        if diff_chunks:
            diff_bytes = b"".join(diff_chunks)
        else:
            diff_bytes = b"".join(
                b" " + line for line in unoptimized_stdout.splitlines(keepends=True)
            )
        ref_path = test.with_suffix(".diff")
        if update:
            ref_path.write_bytes(diff_bytes)
        else:
            expected = ref_path.read_bytes()
            if diff_bytes != expected:
                if run_mod.interrupt_requested():
                    run_mod.report_interrupted_test(test)
                else:
                    run_mod.report_failure(test, "")
                    run_mod.print_diff(expected, diff_bytes, ref_path)
                return False
        run_mod.success(test)
        return True


__all__ = ["DiffRunner"]
