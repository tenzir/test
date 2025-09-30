from __future__ import annotations

import os
from pathlib import Path

from tenzir_test import config, fixtures, run


def test_shell_runner_selected_by_default(tmp_path: Path) -> None:
    script = tmp_path / "tests" / "check.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("echo ok\n", encoding="utf-8")

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        run.refresh_runner_metadata()
        parsed = run.parse_test_config(script)
        assert parsed["runner"] == "shell"
    finally:
        run.apply_settings(original_settings)
        run.refresh_runner_metadata()


def test_shell_runner_executes_with_fixtures(tmp_path: Path) -> None:
    helper = tmp_path / "_shell" / "helper"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("#!/bin/sh\necho helper-ok\n", encoding="utf-8")
    helper.chmod(0o755)

    script_dir = tmp_path / "tests" / "shell"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / "env-check.sh"
    script.write_text(
        """# fixtures: [demo]\n\nset -eu\n\ndir="$(dirname "$0")"\nhelper > "$dir/helper.txt"\nprintf %s "$DEMO_SHELL_FIXTURE" > "$dir/fixture.txt"\nprintf %s "$TENZIR_TMP_DIR" > "$dir/tmp-dir.txt"\n""",
        encoding="utf-8",
    )
    script.chmod(0o755)

    @fixtures.fixture(name="demo", replace=True)
    def _demo_fixture():
        yield {"DEMO_SHELL_FIXTURE": "fixture-ok"}

    original_settings = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )

    try:
        run.apply_settings(
            config.Settings(
                root=tmp_path,
                tenzir_binary=run.TENZIR_BINARY,
                tenzir_node_binary=run.TENZIR_NODE_BINARY,
            )
        )
        parsed = run.parse_test_config(script)
        assert parsed["fixtures"] == ("demo",)
        runner = run.ShellRunner()
        assert runner.run(script, update=False, coverage=False)
        assert runner.run(script, update=True, coverage=False)
        assert (script_dir / "helper.txt").read_text(encoding="utf-8").strip() == "helper-ok"
        assert (script_dir / "fixture.txt").read_text(encoding="utf-8") == "fixture-ok"
        tmp_dir_value = (script_dir / "tmp-dir.txt").read_text(encoding="utf-8").strip()
        assert tmp_dir_value
        assert tmp_dir_value.startswith(str(tmp_path))
    finally:
        run.apply_settings(original_settings)
        fixtures._FACTORIES.pop("demo", None)  # type: ignore[attr-defined]
        run.refresh_runner_metadata()
        if (script_dir / "helper.txt").exists():
            os.remove(script_dir / "helper.txt")
        if (script_dir / "fixture.txt").exists():
            os.remove(script_dir / "fixture.txt")
        if (script_dir / "tmp-dir.txt").exists():
            os.remove(script_dir / "tmp-dir.txt")
