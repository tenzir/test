from __future__ import annotations

import subprocess
import sys
import typing
from pathlib import Path

from tenzir_test import fixtures as fixture_api

from ._utils import get_run_module
from .ext_runner import ExtRunner


class ShellRunner(ExtRunner):
    def __init__(self) -> None:
        super().__init__(name="shell", ext="sh")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        del coverage
        run_mod = get_run_module()
        test_config = run_mod.parse_test_config(test)
        passthrough = run_mod.is_passthrough_enabled()
        inputs_override = typing.cast(str | None, test_config.get("inputs"))
        env, _config_args = run_mod.get_test_env_and_config_args(test, inputs=inputs_override)
        fixtures = typing.cast(tuple[str, ...], test_config.get("fixtures", tuple()))
        timeout = typing.cast(int, test_config["timeout"])
        expect_error = bool(test_config.get("error", False))

        context_token = fixture_api.push_context(
            fixture_api.FixtureContext(
                test=test,
                config=typing.cast(dict[str, typing.Any], test_config),
                coverage=False,
                env=env,
                config_args=tuple(),
                tenzir_binary=run_mod.TENZIR_BINARY,
                tenzir_node_binary=run_mod.TENZIR_NODE_BINARY,
            )
        )
        try:
            with fixture_api.activate(fixtures) as fixture_env:
                env.update(fixture_env)
                run_mod._apply_fixture_env(env, fixtures)
                shell_bin_dir = run_mod.ROOT / "_shell"
                shell_path_prefix = shell_bin_dir.as_posix()
                if env.get("PATH"):
                    env["PATH"] = f"{shell_path_prefix}:{env['PATH']}"
                else:
                    env["PATH"] = shell_path_prefix

                try:
                    completed = run_mod.run_subprocess(
                        ["sh", "-eu", str(test)],
                        env=env,
                        timeout=timeout,
                        capture_output=not passthrough,
                        check=not expect_error,
                        text=False,
                        cwd=str(run_mod.ROOT),
                    )
                except subprocess.CalledProcessError as exc:
                    completed = exc  # treat like CompletedProcess for diagnostics
        finally:
            fixture_api.pop_context(context_token)
            run_mod.cleanup_test_tmp_dir(env.get(run_mod.TEST_TMP_ENV_VAR))

        stdout_data = completed.stdout
        if isinstance(stdout_data, str):
            stdout_bytes: bytes = stdout_data.encode()
        else:
            stdout_bytes = stdout_data or b""

        stderr_data = completed.stderr
        if isinstance(stderr_data, str):
            stderr_bytes: bytes = stderr_data.encode()
        else:
            stderr_bytes = stderr_data or b""

        good = completed.returncode == 0
        if expect_error == good:
            suppressed = run_mod.should_suppress_failure_output()
            summary_line = run_mod.format_failure_message(
                f"got unexpected exit code {completed.returncode}"
            )
            if passthrough:
                if not suppressed:
                    run_mod.report_failure(test, summary_line)
                return False
            if suppressed:
                return False

            with run_mod.stdout_lock:
                run_mod.fail(test)
                line_prefix = "│ ".encode()
                for line in stdout_bytes.splitlines():
                    sys.stdout.buffer.write(line_prefix + line + b"\n")
                if stderr_bytes:
                    sys.stdout.write("├─▶ stderr\n")
                    detail_prefix = run_mod.DETAIL_COLOR.encode()
                    reset_bytes = run_mod.RESET_COLOR.encode()
                    for line in stderr_bytes.splitlines():
                        sys.stdout.buffer.write(
                            line_prefix + detail_prefix + line + reset_bytes + b"\n"
                        )
                sys.stdout.write(summary_line + "\n")
            return False

        if passthrough:
            run_mod.success(test)
            return True

        root_prefix: bytes = (str(run_mod.ROOT) + "/").encode()
        stdout_bytes = stdout_bytes.replace(root_prefix, b"")

        stdout_path = test.with_suffix(".txt")

        combined_bytes = stdout_bytes
        if stderr_bytes:
            if combined_bytes and not combined_bytes.endswith(b"\n"):
                combined_bytes += b"\n"
            combined_bytes += stderr_bytes

        if update:
            stdout_path.write_bytes(combined_bytes)
            run_mod.success(test)
            return True

        if combined_bytes:
            if not stdout_path.exists():
                run_mod.report_failure(
                    test,
                    run_mod.format_failure_message(f'Failed to find ref file: "{stdout_path}"'),
                )
                return False
            run_mod.log_comparison(test, stdout_path, mode="comparing")
            expected_stdout = stdout_path.read_bytes()
            if expected_stdout != combined_bytes:
                if run_mod.interrupt_requested():
                    run_mod.report_interrupted_test(test)
                else:
                    run_mod.report_failure(test, "")
                    run_mod.print_diff(expected_stdout, combined_bytes, stdout_path)
                return False
        elif stdout_path.exists():
            expected_stdout = stdout_path.read_bytes()
            if expected_stdout not in {b"", b"\n"}:
                if run_mod.interrupt_requested():
                    run_mod.report_interrupted_test(test)
                else:
                    run_mod.report_failure(test, "")
                    run_mod.print_diff(expected_stdout, b"", stdout_path)
                return False

        run_mod.success(test)
        return True


__all__ = ["ShellRunner"]
