# Architecture Review

## Summary

The changes add a `--verbose` flag to control per-test output verbosity. The implementation follows existing patterns for global state management (similar to `SHOW_DIFF_OUTPUT` and `SHOW_DIFF_STAT`). The design is clean and integrates well with the existing architecture, though there are minor concerns about coupling between output options.

## Findings

### ARC-1 · P3 · Implicit verbose activation in passthrough mode · 85%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3426`
- **Issue**: The `verbose` flag is silently forced true when passthrough mode is enabled, coupling two orthogonal output behaviors.
- **Reasoning**: The line `set_verbose_output(verbose or passthrough_mode)` creates an implicit dependency where enabling passthrough always enables verbose output. While this may be the desired UX (passthrough implies wanting to see all output), it reduces composability. A user might want passthrough streaming without the skip/success messages cluttering output.
- **Evidence**:

  ```python
  set_verbose_output(verbose or passthrough_mode)
  ```

  This hardcodes the relationship between two flags that conceptually serve different purposes: passthrough controls output streaming, while verbose controls status messages.

- **Suggestion**: Consider making this coupling explicit in the CLI help text for `--passthrough`, e.g., "Stream raw test output directly to the terminal (implies --verbose)." Alternatively, keep them independent and let users compose flags as needed.

### ARC-2 · P4 · Conditional logic asymmetry in summary output · 82%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3699-3706`
- **Issue**: The verbose check only gates `_print_detailed_summary` but not `_print_ascii_summary`, creating asymmetric behavior.
- **Reasoning**: When `summary_enabled` is true but `is_verbose_output()` is false, users get the ASCII summary but not the detailed summary. This makes the output mode partially controlled by verbose, which may confuse users expecting consistent summary output when they explicitly request it via `--show-summary`, `--runner-summary`, or `--fixture-summary`.
- **Evidence**:

  ```python
  summary_enabled = show_summary or runner_summary or fixture_summary
  if summary_enabled:
      if is_verbose_output():
          _print_detailed_summary(project_summary)
      _print_ascii_summary(
          project_summary,
          include_runner=runner_summary,
          include_fixture=fixture_summary,
      )
  ```

  The detailed summary requires verbose, but the ASCII summary does not.

- **Suggestion**: Document this behavior clearly, or consider making the detailed summary always appear when explicitly requested via summary flags, regardless of verbose mode. The purpose of verbose appears to be about per-test output, not summaries.
