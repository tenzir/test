# UX Review

## Summary

The changes introduce a `--verbose` flag that enables fine-grained test result reporting. Overall, the implementation maintains good UX patterns consistent with the existing codebase by following established conventions for option flags and global state management. The feature integrates smoothly with passthrough mode and summary display logic. Minor discoverability issues exist around automatic verbose enablement and the help text's implicit coupling with passthrough mode.

## Findings

### UXD-1 · P3 · Implicit verbose enablement in passthrough mode unclear in help text · 78%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/cli.py:136-141`
- **Issue**: The `--verbose` help text does not indicate that verbose output is automatically enabled in passthrough mode
- **Reasoning**: Users enabling passthrough mode with `--passthrough` will automatically get verbose output without explicitly requesting it (via `set_verbose_output(verbose or passthrough_mode)` at line 3426 of run.py). The help text for `--verbose` only states "Print individual test results as they complete" without mentioning this automatic coupling, which could confuse users when verbose output appears without the flag.
- **Evidence**:
  - Help text: `"Print individual test results as they complete."`
  - Implementation at run.py:3426: `set_verbose_output(verbose or passthrough_mode)`
  - This creates an implicit dependency where `--passthrough` triggers `--verbose` behavior without the user being aware through documentation.
- **Suggestion**: Update the help text to clarify the behavior: `"Print individual test results as they complete (automatically enabled in passthrough mode)."` This sets proper expectations and aids discoverability.

### UXD-2 · P3 · Silent behavior change for skip messages with verbose flag disabled · 75%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3042-3046`
- **Issue**: Skip reason messages no longer print by default; they only appear when `--verbose` is explicitly enabled
- **Reasoning**: Previously, skip messages were always printed. Now they're conditional on `is_verbose_output()`. Users who expect skip reasons to be visible for test filtering and validation purposes will see no output by default, potentially affecting their workflow. The behavior change is silent—no feedback to indicate messages are being suppressed.
- **Evidence**:

  ```python
  def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
      if is_verbose_output():  # NEW: conditional on verbose flag
          rel_path = _relativize_path(test)
          suite_suffix = _format_suite_suffix()
          print(f"{SKIP} skipped {rel_path}{suite_suffix}: {reason}")
  ```

  Before the change, the print statement executed unconditionally. Now it requires the verbose flag.

- **Suggestion**: Either (1) restore skip messages to be always visible since they're informational and support test understanding, or (2) add a dedicated `--show-skips` flag for explicit control, or (3) document this behavior change prominently in release notes so users are aware that skip reasons require `--verbose`.

### UXD-3 · P2 · No feedback when verbose flag has no effect in normal mode · 82%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:2719-2726`
- **Issue**: The `success()` function silently ignores calls when verbose mode is off, providing no feedback to users that the flag they enabled is having an effect
- **Reasoning**: When a user enables `--verbose` but isn't in passthrough mode, they see individual success messages printed by the `success()` function. However, if they then disable the flag, the function returns early with no indication that reporting is suppressed. This creates a discoverability problem: users cannot confirm the flag is working since there's no status message like "verbose mode: on/off" in the startup output.
- **Evidence**:

  ```python
  def success(test: Path) -> None:
      if not is_verbose_output():
          return  # Silent return, no feedback that verbose is disabled
      with stdout_lock:
          rel_test = _relativize_path(test)
          suite_suffix = _format_suite_suffix()
          attempt_suffix = _format_attempt_suffix()
          print(f"{CHECKMARK} {rel_test}{suite_suffix}{attempt_suffix}")
  ```

  The early return provides zero indication that verbose output is disabled.

- **Suggestion**: Add startup feedback when verbose mode is enabled. For example, after line 3426 in run_cli where `set_verbose_output()` is called, consider printing: `print(f"{INFO} verbose output enabled")` if verbose mode was explicitly set (not just auto-enabled by passthrough). This helps users confirm the flag is working.

### UXD-4 · P3 · Inconsistent verbosity between success messages and detailed summary · 80%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3699-3706`
- **Issue**: Success messages print when verbose is enabled (line 2720), but detailed summary only prints when verbose is enabled AND summary output is requested (line 3700)
- **Reasoning**: A user enabling `--verbose` will see success messages for individual tests but may not see the detailed summary breakdown of failures if they forget to add `--summary`/`--fixture-summary`/`--runner-summary`. The conditional at line 3700 creates dual requirements that could confuse users: verbose output for individual tests works alone, but detailed failure summaries require both verbose AND a summary flag.
- **Evidence**:

  ```python
  # Line 2720: success prints when verbose is on
  if not is_verbose_output():
      return

  # Line 3700: detailed summary prints when verbose AND summary flags are on
  summary_enabled = show_summary or runner_summary or fixture_summary
  if summary_enabled:
      if is_verbose_output():
          _print_detailed_summary(project_summary)
  ```

  The asymmetry between individual success verbosity and summary verbosity could lead to confusion.

- **Suggestion**: Document that `--verbose` controls per-test feedback and that `--summary`/`--runner-summary`/`--fixture-summary` control aggregate reporting. Alternatively, consider making detailed summary print whenever `--verbose` is enabled (remove the `summary_enabled` guard), since detailed failure info is more valuable when verbose mode is requested.

### UXD-5 · P4 · Verbose flag not documented in primary help output discovery path · 70%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/cli.py:136-141`
- **Issue**: The `--verbose` flag appears between `--passthrough` and `--all-projects` in the decorator stack, which may affect help text order visibility for users
- **Reasoning**: Click options are displayed in help text in the order they're decorated (bottom-to-top visually). The `--verbose` flag is positioned late in the decorator chain, which may cause it to appear lower in the help output, potentially reducing discoverability. Users scrolling through help text might miss the feature.
- **Evidence**:
  - Option is placed after `--passthrough` (line 130-135) and before `--all-projects` (line 142-147)
  - Help text shows flags in decorator order, so late placement = lower visibility
- **Suggestion**: This is a minor cosmetic issue. Consider documenting the feature in any project-level README or docs to ensure discovery. The current placement is not wrong, just less prominent than `-p`/`--passthrough` above it.

## Confidence Score Methodology

- **UXD-1**: Moderate confidence (78%) - The implicit coupling is real and documented in code, but users relying on passthrough typically expect verbose output, so the UX impact is mitigated by reasonable expectations.
- **UXD-2**: Moderate-high confidence (82%) - The behavioral change is definitively silent (no print, no log message), but skip messages were already optional feature, so severity is reduced.
- **UXD-3**: High confidence (82%) - No startup feedback mechanism exists, and users have no way to confirm the flag is active without examining output changes test-to-test.
- **UXD-4**: Moderate-high confidence (80%) - The inconsistency is clear in code (two different guard conditions), though the UX impact is mitigated if users know to use both flags together.
- **UXD-5**: Low-moderate confidence (70%) - Placement affects discoverability but doesn't prevent access; users can find it via `--help` search or documentation.
