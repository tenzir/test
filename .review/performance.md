# Performance Review

## Summary

The changes introduce a verbose output flag that conditionally skips per-test output and detailed summary printing. The implementation is sound with good performance characteristics - adding early-return guards prevents unnecessary string formatting and I/O operations when verbose mode is disabled. No significant performance concerns were identified.

## Findings

### PRF-1 · P4 · Path relativization called unconditionally in verbose-guarded code · 82%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3044-3046`
- **Issue**: `_relativize_path()` and `_format_suite_suffix()` are called before checking the verbose flag in `handle_skip()`
- **Reasoning**: The verbose check happens at line 3043, but the expensive formatting operations occur at lines 3044-3045 inside the conditional block. This is actually correct - the operations only execute when verbose is enabled. However, examining the change more carefully shows these calls are properly guarded by the `if is_verbose_output():` check, so they won't execute when verbose is disabled.
- **Evidence**:

  ```python
  def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
      if is_verbose_output():
          rel_path = _relativize_path(test)
          suite_suffix = _format_suite_suffix()
          print(f"{SKIP} skipped {rel_path}{suite_suffix}: {reason}")
  ```

  The formatting calls are inside the conditional, so they're properly guarded.

- **Suggestion**: No action needed - the code is already optimized correctly.

### PRF-2 · P4 · Early return optimization applied correctly in success() · 95%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:2719-2721`
- **Issue**: Not an issue - this is a positive finding
- **Reasoning**: The `success()` function now checks `is_verbose_output()` and returns early before acquiring the stdout lock or performing any string operations. This is an optimal implementation that saves both computation and lock contention when verbose mode is disabled.
- **Evidence**:

  ```python
  def success(test: Path) -> None:
      if not is_verbose_output():
          return
      with stdout_lock:
          rel_test = _relativize_path(test)
          suite_suffix = _format_suite_suffix()
          attempt_suffix = _format_attempt_suffix()
          print(f"{CHECKMARK} {rel_test}{suite_suffix}{attempt_suffix}")
  ```

  The early return at line 2720-2721 prevents:
  - Lock acquisition (line 2722)
  - Three function calls for path/suffix formatting (lines 2723-2725)
  - String formatting and I/O (line 2726)

- **Suggestion**: This is well-implemented. Consider documenting this pattern for other output functions.

### PRF-3 · P3 · Detailed summary tree building avoided when not verbose · 88%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3700-3701`
- **Issue**: The detailed summary printing is now conditionally skipped when verbose mode is disabled
- **Reasoning**: `_print_detailed_summary()` calls `_build_path_tree()` which sorts paths and builds a nested dictionary structure, then calls `_render_tree()` to recursively render it. When tests number in the hundreds, this involves O(n log n) sorting plus tree construction. The change at line 3700 now skips this work entirely when verbose is disabled, but keeps the ASCII summary which is always shown when summary is enabled.
- **Evidence**:

  ```python
  if summary_enabled:
      if is_verbose_output():
          _print_detailed_summary(project_summary)
      _print_ascii_summary(
          project_summary,
          include_runner=runner_summary,
          include_fixture=fixture_summary,
      )
  ```

  The `_print_detailed_summary()` implementation at line 2685-2699 shows:

  ```python
  def _print_detailed_summary(summary: Summary) -> None:
      if not summary.failed_paths and not summary.skipped_paths:
          return
      # ... builds tree and renders it
      for line in _render_tree(_build_path_tree(summary.skipped_paths)):
          print(f"  {line}")
  ```

- **Suggestion**: This optimization is appropriate. The detailed tree view is useful in verbose mode but unnecessary overhead otherwise since the ASCII summary provides aggregate statistics.

## Positive Observations

1. **Lock contention reduced**: The early return in `success()` (line 2720) prevents unnecessary lock acquisition in the common case where tests pass and verbose mode is disabled. Since `success()` is called once per passing test, this reduces lock contention in multi-threaded test execution.

2. **Per-test overhead eliminated**: Both `success()` and `handle_skip()` now avoid formatting operations (path relativization, suffix formatting) when verbose is disabled, eliminating O(1) work per test that scales linearly with test count.

3. **Global flag pattern**: The implementation uses a module-level `VERBOSE_OUTPUT` flag accessed via `is_verbose_output()`, which is more efficient than passing parameters through the call stack or checking CLI args repeatedly.

4. **Passthrough mode integration**: Line 3424 shows `set_verbose_output(verbose or passthrough_mode)` - passthrough mode automatically enables verbose output, which is semantically correct since passthrough needs to show all output.
