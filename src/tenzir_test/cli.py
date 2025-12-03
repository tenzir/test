"""Command line entry point for the tenzir-test runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import click

from . import __version__, run as runtime


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
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="tenzir-test",
    message="%(prog)s %(version)s",
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
@click.option(
    "package_dirs",
    "--package-dirs",
    multiple=True,
    help=(
        "Comma-separated list of package directories to load (repeatable). "
        "These only control package visibility; test selection still follows the usual --root/args."
    ),
)
@click.argument(
    "tests",
    nargs=-1,
    type=click.Path(path_type=Path, resolve_path=False),
)
@click.option("-u", "--update", is_flag=True, help="Update reference outputs.")
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="Enable debug logging.",
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
    "--summary",
    "show_summary",
    is_flag=True,
    help="Show an aggregate table and detailed failure summary after execution.",
)
@click.option(
    "--diff/--no-diff",
    "show_diff_output",
    default=True,
    help="Show unified diffs when expectations differ.",
)
@click.option(
    "--diff-stat/--no-diff-stat",
    "show_diff_stat",
    default=True,
    help="Include per-file diff statistics and change counters when expectations differ.",
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
@click.option(
    "-p",
    "--passthrough",
    is_flag=True,
    help="Stream raw test output directly to the terminal.",
)
@click.option(
    "-a",
    "--all-projects",
    is_flag=True,
    help="Run the root project alongside any selected satellites.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    *,
    root: Path | None,
    tenzir_binary: Path | None,
    tenzir_node_binary: Path | None,
    package_dirs: tuple[str, ...],
    tests: tuple[Path, ...],
    update: bool,
    debug: bool,
    purge: bool,
    coverage: bool,
    coverage_source_dir: Path | None,
    runner_summary: bool,
    fixture_summary: bool,
    show_summary: bool,
    show_diff_output: bool,
    show_diff_stat: bool,
    keep_tmp_dirs: bool,
    jobs: int,
    passthrough: bool,
    all_projects: bool,
) -> int:
    """Execute tenzir-test scenarios."""

    package_paths: list[Path] = []
    for entry in package_dirs:
        for piece in entry.split(","):
            piece = piece.strip()
            if not piece:
                continue
            package_paths.append(Path(piece))

    jobs_source = ctx.get_parameter_source("jobs")
    jobs_overridden = jobs_source is not click.core.ParameterSource.DEFAULT

    try:
        result = runtime.run_cli(
            root=root,
            tenzir_binary=tenzir_binary,
            tenzir_node_binary=tenzir_node_binary,
            package_dirs=package_paths,
            tests=list(tests),
            update=update,
            debug=debug,
            purge=purge,
            coverage=coverage,
            coverage_source_dir=coverage_source_dir,
            runner_summary=runner_summary,
            fixture_summary=fixture_summary,
            show_summary=show_summary,
            show_diff_output=show_diff_output,
            show_diff_stat=show_diff_stat,
            keep_tmp_dirs=keep_tmp_dirs,
            jobs=jobs,
            passthrough=passthrough,
            jobs_overridden=jobs_overridden,
            all_projects=all_projects,
        )
    except runtime.HarnessError as exc:
        if exc.show_message and exc.args:
            raise click.ClickException(str(exc)) from exc
        return exc.exit_code
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Click command and translate Click exits to integer codes."""

    command_main = getattr(cli, "main")
    previous_color_mode = runtime.get_color_mode()
    runtime.set_color_mode(runtime.ColorMode.AUTO)
    try:
        result = command_main(
            args=list(argv) if argv is not None else None,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:  # pragma: no cover - passthrough CLI termination
        return _normalize_exit_code(exc.exit_code)
    except click.exceptions.ClickException as exc:
        exc.show(file=sys.stderr)
        exit_code = getattr(exc, "exit_code", None)
        return _normalize_exit_code(exit_code)
    except SystemExit as exc:  # pragma: no cover - propagate runner exits
        return _normalize_exit_code(exc.code)
    else:
        return _normalize_exit_code(result)
    finally:
        runtime.set_color_mode(previous_color_mode)


if __name__ == "__main__":
    raise SystemExit(main())
