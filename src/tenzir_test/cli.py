"""Command line entry point for the tenzir-test runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from . import run as runtime


def _normalize_exit_code(value: object) -> int:
    """Cast arbitrary exit codes to integers."""

    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "root",
    "--root",
    type=click.Path(
        path_type=Path, file_okay=False, dir_okay=True, writable=False, resolve_path=False
    ),
    help="Project root to scan for tests.",
)
@click.option(
    "tenzir_binary",
    "--tenzir-binary",
    type=click.Path(path_type=Path, dir_okay=False, writable=False, resolve_path=False),
    help="Path to the tenzir executable.",
)
@click.option(
    "tenzir_node_binary",
    "--tenzir-node-binary",
    type=click.Path(path_type=Path, dir_okay=False, writable=False, resolve_path=False),
    help="Path to the tenzir-node executable.",
)
@click.argument(
    "tests",
    nargs=-1,
    type=click.Path(path_type=Path, resolve_path=False),
)
@click.option("-u", "--update", is_flag=True, help="Update reference outputs.")
@click.option(
    "-v",
    "--log-comparisons",
    is_flag=True,
    help="Log reference comparison activity.",
)
@click.option("--purge", is_flag=True, help="Delete cached runner artifacts and exit.")
@click.option(
    "--coverage",
    is_flag=True,
    help="Enable code coverage collection (increases timeouts by 5x).",
)
@click.option(
    "--coverage-source-dir",
    type=click.Path(path_type=Path, resolve_path=False),
    help="Source directory for coverage path mapping (defaults to current directory).",
)
@click.option(
    "--runner-summary",
    is_flag=True,
    help="Include per-runner statistics in the summary table.",
)
@click.option(
    "--fixture-summary",
    is_flag=True,
    help="Include per-fixture statistics in the summary table.",
)
@click.option(
    "-k",
    "--keep",
    "keep_tmp_dirs",
    is_flag=True,
    help="Preserve per-test temporary directories instead of deleting them.",
)
@click.option(
    "-j",
    "--jobs",
    type=click.IntRange(min=1),
    default=runtime.get_default_jobs(),
    show_default=True,
    metavar="N",
    help="Number of parallel worker threads.",
)
def cli(
    *,
    root: Path | None,
    tenzir_binary: Path | None,
    tenzir_node_binary: Path | None,
    tests: tuple[Path, ...],
    update: bool,
    log_comparisons: bool,
    purge: bool,
    coverage: bool,
    coverage_source_dir: Path | None,
    runner_summary: bool,
    fixture_summary: bool,
    keep_tmp_dirs: bool,
    jobs: int,
) -> None:
    """Execute tenzir-test scenarios."""

    runtime.run_cli(
        root=root,
        tenzir_binary=tenzir_binary,
        tenzir_node_binary=tenzir_node_binary,
        tests=list(tests),
        update=update,
        log_comparisons=log_comparisons,
        purge=purge,
        coverage=coverage,
        coverage_source_dir=coverage_source_dir,
        runner_summary=runner_summary,
        fixture_summary=fixture_summary,
        keep_tmp_dirs=keep_tmp_dirs,
        jobs=jobs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Click command and translate Click exits to integer codes."""

    command_main = getattr(cli, "main")
    try:
        command_main(
            args=list(argv) if argv is not None else None,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:  # pragma: no cover - passthrough CLI termination
        return _normalize_exit_code(exc.exit_code)
    except SystemExit as exc:  # pragma: no cover - propagate runner exits
        return _normalize_exit_code(exc.code)
    except click.exceptions.Abort:  # pragma: no cover - propagate Ctrl+C
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
