from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tenzir_test import runners
from tenzir_test.runners._utils import get_run_module


class XxdRunner(runners.ExtRunner):
    """Dump hex output for ``*.xxd`` tests using the ``xxd`` utility."""

    def __init__(self) -> None:
        super().__init__(name="xxd", ext="xxd")

    def run(self, test: Path, update: bool, coverage: bool = False) -> bool:
        del coverage
        run_mod = get_run_module()
        try:
            completed = subprocess.run(
                ["xxd", "-g1", str(test)],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            run_mod.report_failure(
                test,
                "└─▶ \033[31mxxd is not available on PATH\033[0m",
            )
            return False
        if completed.returncode != 0:
            with run_mod.stdout_lock:
                run_mod.fail(test)
                print(
                    f"└─▶ \033[31mxxd exited with {completed.returncode}\033[0m",
                    file=sys.stdout,
                )
                if completed.stdout:
                    sys.stdout.buffer.write(completed.stdout)
                if completed.stderr:
                    sys.stdout.buffer.write(completed.stderr)
            return False

        output = completed.stdout
        ref_path = test.with_suffix(".txt")
        if update:
            ref_path.write_bytes(output)
            run_mod.success(test)
            return True
        if not ref_path.exists():
            run_mod.report_failure(
                test,
                f'└─▶ \033[31mMissing reference file "{ref_path}"\033[0m',
            )
            return False
        run_mod.log_comparison(test, ref_path, mode="comparing")
        expected = ref_path.read_bytes()
        if expected != output:
            run_mod.report_failure(test, "")
            run_mod.print_diff(expected, output, ref_path)
            return False
        run_mod.success(test)
        return True


runners.register(XxdRunner())
