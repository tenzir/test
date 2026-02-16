"""Command line entry point for the tenzir-test runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import click

from . import __version__, run as runtime


class _UnindentedEpilogCommand(click.Command):
    """Command that renders the epilog without indentation."""

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write(self.epilog)


def _normalize_exit_code(value: object) -> int:
    """Cast arbitrary exit codes to integers."""

    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


@click.command(
    cls=_UnindentedEpilogCommand,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="""\
Examples:
  tenzir-test                          Run all tests in the project
  tenzir-test tests/alerts/            Run all tests in a directory
  tenzir-test tests/basic.tql          Run a specific test file
  tenzir-test -m '*ctx*'               Run tests matching a path pattern
  tenzir-test -m '*add*' -m '*del*'    Run tests matching either pattern
  tenzir-test tests/ctx/ -m '*create*' Intersect directory with pattern
  tenzir-test --run-skipped            Run all skipped tests unconditionally
  tenzir-test --run-skipped-reason maintenance
                                       Run only skipped tests whose reason matches
  tenzir-test -u tests/new.tql         Run test and update its baseline
  tenzir-test -p -k tests/debug/       Debug with output streaming and kept temps
  tenzir-test --fixture mysql          Start fixture(s) in foreground mode
  tenzir-test --fixture 'kafka: {port: 9092}' --debug

Documentation: https://docs.tenzir.com/reference/test-framework/
""",
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
    "package_dirs",
    "--package-dirs",
    multiple=True,
    help=(
        "Comma-separated list of package directories to load (repeatable). "
        "These only control package visibility; test selection still follows the usual --root/args."
    ),
)
@click.option(
    "fixtures",
    "--fixture",
    multiple=True,
    help=(
        "Activate fixtures in standalone foreground mode (repeatable). "
        "Accepts bare names ('mysql') or mapping specs "
        "('kafka: {port: 9092}'). "
        "When provided, positional TEST arguments are not allowed."
    ),
)
@click.argument(
    "tests",
    nargs=-1,
    metavar="[TEST]...",
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
    "-v",
    "--verbose",
    is_flag=True,
    help="Print individual test results as they complete. By default, only failures are shown. Automatically enabled in passthrough mode.",
)
@click.option(
    "--run-skipped",
    is_flag=True,
    help="Run all skipped tests unconditionally.",
)
@click.option(
    "--run-skipped-reason",
    "run_skipped_reasons",
    multiple=True,
    type=str,
    help=(
        "Run tests skipped for reasons matching a substring or glob pattern. "
        "Bare strings match as substrings; glob metacharacters (*, ?, [) use fnmatch mode. "
        "Repeatable; matching any reason pattern selects the skipped test. "
        "Matches against the final displayed skip reason."
    ),
)
@click.option(
    "-a",
    "--all-projects",
    is_flag=True,
    help="Run the root project alongside any selected satellites.",
)
@click.option(
    "-m",
    "--match",
    "match_patterns",
    multiple=True,
    type=str,
    help=(
        "Run tests whose relative path matches a substring or glob pattern. "
        "Bare strings match as substrings (e.g. 'mysql' matches any path "
        "containing 'mysql'). Glob metacharacters (*, ?, [) trigger fnmatch "
        "mode. Repeatable; tests matching any pattern are selected. "
        "If TEST paths are also given, only tests matching both the path and "
        "pattern are run (intersection). "
        "Note: if a matched test belongs to a suite (configured via test.yaml), "
        "all tests in that suite are included automatically."
    ),
)
@click.pass_context
def cli(
    ctx: click.Context,
    *,
    root: Path | None,
    package_dirs: tuple[str, ...],
    fixtures: tuple[str, ...],
    tests: tuple[Path, ...],
    match_patterns: tuple[str, ...],
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
    verbose: bool,
    run_skipped: bool,
    run_skipped_reasons: tuple[str, ...],
    all_projects: bool,
) -> int:
    """Execute test scenarios and compare output against baselines.

    Discovers and runs tests under the project root, comparing actual output
    against reference .txt files. Use --update to regenerate baselines.

    \b
    TEST paths can be:
      - Individual test files (e.g., tests/basic.tql)
      - Directories to run all tests within (e.g., tests/alerts/)
      - Omitted to run all discovered tests in the project

    Use --fixture to start fixtures without executing tests. This mode keeps
    fixtures running in the foreground until interrupted.

    Use -m/--match to select tests by substring or glob pattern.
    Bare strings match as substrings; glob metacharacters (*, ?, [) trigger
    fnmatch mode. Patterns match against relative paths shown in test output.
    When both TEST paths and -m patterns are given, only tests matching both
    are run (intersection). Empty pattern strings are silently ignored.
    If a matched test belongs to a suite (configured via test.yaml), all
    tests in that suite are included automatically.
    """

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
        if fixtures:
            if tests:
                raise click.UsageError(
                    "positional TEST arguments cannot be used with --fixture mode"
                )
            return runtime.run_fixture_mode_cli(
                root=root,
                package_dirs=package_paths,
                fixtures=list(fixtures),
                debug=debug,
                keep_tmp_dirs=keep_tmp_dirs,
            )

        result = runtime.run_cli(
            root=root,
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
            verbose=verbose,
            run_skipped=run_skipped,
            run_skipped_reasons=list(run_skipped_reasons),
            jobs_overridden=jobs_overridden,
            all_projects=all_projects,
            match_patterns=list(match_patterns),
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
