from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import subprocess
import sys

import pytest

from tenzir_test import config, run
from tenzir_test.runners import TenzirRunner


@pytest.fixture()
def configured_root(tmp_path: Path) -> Iterator[Path]:
    original = config.Settings(
        root=run.ROOT,
        tenzir_binary=run.TENZIR_BINARY,
        tenzir_node_binary=run.TENZIR_NODE_BINARY,
    )
    run.apply_settings(
        config.Settings(root=tmp_path, tenzir_binary=(sys.executable,), tenzir_node_binary=None)
    )
    try:
        yield tmp_path
    finally:
        run.apply_settings(original)


def test_quiet_defaults_to_false(configured_root: Path) -> None:
    test = configured_root / "case.tql"
    test.write_text("from {x: 1}\n", encoding="utf-8")

    assert run.parse_test_config(test)["quiet"] is False


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_quiet_frontmatter(configured_root: Path, value: str, expected: bool) -> None:
    test = configured_root / "case.tql"
    test.write_text(f"---\nquiet: {value}\n---\nfrom {{x: 1}}\n", encoding="utf-8")

    assert run.parse_test_config(test)["quiet"] is expected


@pytest.mark.parametrize("origin", ["test", "directory"])
@pytest.mark.parametrize("value", ["1", "null", "sometimes", "[]", "{}"])
def test_quiet_rejects_invalid_values(configured_root: Path, origin: str, value: str) -> None:
    test = configured_root / "case.tql"
    if origin == "directory":
        (configured_root / "test.yaml").write_text(f"quiet: {value}\n", encoding="utf-8")
        test.write_text("from {x: 1}\n", encoding="utf-8")
    else:
        test.write_text(f"---\nquiet: {value}\n---\nfrom {{x: 1}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid value for 'quiet'"):
        run.parse_test_config(test)


def test_quiet_directory_inheritance_and_overrides(configured_root: Path) -> None:
    (configured_root / "test.yaml").write_text("quiet: true\n", encoding="utf-8")
    nested = configured_root / "nested"
    nested.mkdir()
    inherited = nested / "inherited.tql"
    inherited.write_text("from {x: 1}\n", encoding="utf-8")
    overridden = nested / "overridden.tql"
    overridden.write_text("---\nquiet: false\n---\nfrom {x: 1}\n", encoding="utf-8")
    deeper = nested / "deeper"
    deeper.mkdir()
    (deeper / "test.yaml").write_text("quiet: false\n", encoding="utf-8")
    directory_override = deeper / "case.tql"
    directory_override.write_text("from {x: 1}\n", encoding="utf-8")

    assert run.parse_test_config(inherited)["quiet"] is True
    assert run.parse_test_config(overridden)["quiet"] is False
    assert run.parse_test_config(directory_override)["quiet"] is False


@pytest.mark.parametrize(
    "header",
    [
        "",
        "\n",
        "---\nquiet: true\n---\n",
        "---\nquiet: true\n---\n// parallelism: disabled\n\n",
        "#!/usr/bin/env tenzir\n",
    ],
)
@pytest.mark.parametrize("trailing_newline", ["", "\n"])
def test_prepare_quiet_test_preserves_metadata_and_pipeline(
    tmp_path: Path, header: str, trailing_newline: str
) -> None:
    test = tmp_path / "case.tql"
    source = f'{header}from {{x: "bad"}}\nx = x.parse_time("%Y-%m-%d") // comment{trailing_newline}'
    test.write_text(source, encoding="utf-8")
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()

    wrapped = run._prepare_quiet_test(test, wrapper_dir)

    assert wrapped == wrapper_dir / test.name
    assert wrapped.read_text(encoding="utf-8") == (
        f'{header}quiet {{\nfrom {{x: "bad"}}\nx = x.parse_time("%Y-%m-%d") // comment\n}}\n'
    )
    assert test.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("source", ["", "\n", "// comment", "---\nquiet: true\n---\n"])
def test_quiet_does_not_wrap_empty_pipelines(tmp_path: Path, source: str) -> None:
    test = tmp_path / "case.tql"
    test.write_text(source, encoding="utf-8")
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()

    assert run._prepare_quiet_test(test, wrapper_dir) == test
    assert not list(wrapper_dir.iterdir())


@pytest.mark.parametrize("quiet", [False, True])
def test_tenzir_runner_quiet_execution(
    monkeypatch: pytest.MonkeyPatch, configured_root: Path, quiet: bool
) -> None:
    test = configured_root / "case.tql"
    source = f"---\nquiet: {str(quiet).lower()}\n---\nfrom {{x: 1}}\n"
    test.write_text(source, encoding="utf-8")
    paths: list[Path] = []

    def execute(command, **kwargs):
        path = Path(command[-1])
        paths.append(path)
        assert command[-2] == "-f"
        assert "--console-verbosity=warning" in command
        assert kwargs["capture_output"] is True
        if quiet:
            assert path != test
            assert path.read_text(encoding="utf-8") == (
                "---\nquiet: true\n---\nquiet {\nfrom {x: 1}\n}\n"
            )
        else:
            assert path == test
        # Compilation diagnostics remain visible and reference the original test,
        # not the temporary wrapper. No stderr filtering takes place.
        stderr = f"warning: compilation diagnostic\n --> {path}:5:1\n".encode()
        return subprocess.CompletedProcess(command, 0, b"events\n", stderr)

    monkeypatch.setattr(run, "run_subprocess", execute)
    runner = TenzirRunner()

    assert runner.run(test, update=True) is True
    assert runner.run(test, update=False) is True
    assert test.with_suffix(".txt").read_bytes() == (
        b"warning: compilation diagnostic\n --> case.tql:5:1\nevents\n"
    )
    assert test.read_text(encoding="utf-8") == source
    if quiet:
        assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("expect_error", [False, True])
def test_quiet_keeps_errors(
    monkeypatch: pytest.MonkeyPatch,
    configured_root: Path,
    capsys: pytest.CaptureFixture[str],
    expect_error: bool,
) -> None:
    test = configured_root / "case.tql"
    test.write_text(
        f"---\nquiet: true\nerror: {str(expect_error).lower()}\n---\nfail\n",
        encoding="utf-8",
    )
    paths: list[Path] = []

    def execute(command, **kwargs):
        path = Path(command[-1])
        paths.append(path)
        stderr = f"error: expected failure\n --> {path}:6:1\n".encode()
        return subprocess.CompletedProcess(command, 1, b"", stderr)

    monkeypatch.setattr(run, "run_subprocess", execute)

    assert TenzirRunner().run(test, update=True) is expect_error
    if expect_error:
        assert test.with_suffix(".txt").read_bytes() == (
            b"error: expected failure\n --> case.tql:6:1\n"
        )
    else:
        assert "error: expected failure" in capsys.readouterr().out
        assert not test.with_suffix(".txt").exists()
    assert all(not path.exists() for path in paths)


def test_quiet_cleans_up_wrapper_on_timeout(
    monkeypatch: pytest.MonkeyPatch, configured_root: Path
) -> None:
    test = configured_root / "case.tql"
    test.write_text("---\nquiet: true\n---\nfrom {x: 1}\n", encoding="utf-8")
    paths: list[Path] = []

    def execute(command, **kwargs):
        path = Path(command[-1])
        paths.append(path)
        assert path.exists()
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(run, "run_subprocess", execute)

    assert TenzirRunner().run(test, update=True) is False
    assert all(not path.exists() for path in paths)


def test_quiet_passthrough_uses_scope_without_capturing(
    monkeypatch: pytest.MonkeyPatch, configured_root: Path
) -> None:
    test = configured_root / "case.tql"
    test.write_text("---\nquiet: true\n---\nfrom {x: 1}\n", encoding="utf-8")
    paths: list[Path] = []

    def execute(command, **kwargs):
        path = Path(command[-1])
        paths.append(path)
        assert "quiet {" in path.read_text(encoding="utf-8")
        assert kwargs["capture_output"] is False
        return subprocess.CompletedProcess(command, 0, None, None)

    monkeypatch.setattr(run, "run_subprocess", execute)
    previous_mode = run.is_passthrough_enabled()
    run.set_passthrough_enabled(True)
    try:
        assert TenzirRunner().run(test, update=False) is True
    finally:
        run.set_passthrough_enabled(previous_mode)

    assert not test.with_suffix(".txt").exists()
    assert all(not path.exists() for path in paths)
