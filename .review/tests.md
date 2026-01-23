# Test Coverage Review

## Summary

The changes introduce a `--verbose` / `-v` flag to control verbose output during test execution. Three new functions handle global state management for this feature, and tests verify the correct behavior with proper state restoration. The changes include adequate test coverage for the new functionality, though edge cases around state management isolation deserve attention.

## Findings

### TST-1 · P2 · Missing test for verbose flag disabling detailed summary · 84%

- **File**: `tests/test_run.py:309-344`
- **Issue**: Tests verify verbose mode enables output, but no test covers that verbose=False suppresses output
- **Reasoning**: The implementation at `src/tenzir_test/run.py:3700` gates `_print_detailed_summary()` on `is_verbose_output()`. The test file provides `test_success_includes_suite_suffix` and `test_handle_skip_uses_skip_glyph` which explicitly set `run.set_verbose_output(True)` before testing output, but there is no corresponding test that verifies the output is suppressed when verbose is False. This is a behavioral asymmetry that could hide bugs where the flag doesn't properly suppress output.
- **Evidence**:

  ```python
  # tests/test_run.py:310-326 - Sets verbose to True explicitly
  run.set_verbose_output(True)
  test_path = tmp_path / "tests" / "example.tql"
  test_path.parent.mkdir(parents=True)
  test_path.touch()
  result = run.handle_skip("slow", test_path, update=False, output_ext="txt")
  assert result == "skipped"
  output = capsys.readouterr().out.strip()
  assert output == f"{run.SKIP} skipped tests/example.tql: slow"
  ```

  But there is no test like:

  ```python
  run.set_verbose_output(False)
  run.handle_skip(...)
  output = capsys.readouterr().out.strip()
  assert output == ""  # Should be empty when verbose=False
  ```

- **Suggestion**: Add two test cases: `test_handle_skip_suppressed_when_verbose_disabled()` and `test_success_suppressed_when_verbose_disabled()` that verify no output is printed when verbose is False.

### TST-2 · P2 · Missing test for verbose mode integration with passthrough mode · 81%

- **File**: `src/tenzir_test/run.py:3426`
- **Issue**: Verbose mode is automatically enabled in passthrough mode, but no test verifies this behavior
- **Reasoning**: At line 3426 of run.py, the code sets `set_verbose_output(verbose or passthrough_mode)`, meaning verbose output is forced on when passthrough is enabled. The existing tests do not verify this automatic activation. Without this test, a future change could accidentally disable verbose output in passthrough mode, breaking the expected behavior.
- **Evidence**:

  ```python
  # src/tenzir_test/run.py:3425-3426
  passthrough_mode = harness_mode is HarnessMode.PASSTHROUGH
  set_verbose_output(verbose or passthrough_mode)
  ```

  Test file has `test_worker_prints_passthrough_header()` but it doesn't verify that verbose output is activated for passthrough tests.

- **Suggestion**: Add test `test_passthrough_mode_enables_verbose_output()` that calls `run_cli()` with `passthrough=True, verbose=False` and verifies that `is_verbose_output()` returns True. Alternatively, extend `test_worker_prints_passthrough_header()` to verify verbose output state.

### TST-3 · P3 · Test isolation not guaranteed for VERBOSE_OUTPUT global state · 78%

- **File**: `tests/test_run.py:309-344`
- **Issue**: Tests save and restore VERBOSE_OUTPUT state, but cleanup is not atomic or guaranteed on exception
- **Reasoning**: The pattern used in tests (save original, run test, restore in finally block) is correct but could be fragile if multiple tests run in parallel without proper isolation. While the try/finally structure is sound for single-threaded test execution, the global state mutation creates implicit coupling between tests. If a test raises an exception before reaching the finally block's restore, or if pytest's test isolation is weak, subsequent tests could see stale state.
- **Evidence**:

  ```python
  # tests/test_run.py:310-326
  original_verbose = run.VERBOSE_OUTPUT
  try:
      run.ROOT = tmp_path
      run.set_verbose_output(True)
      # test code
  finally:
      run.ROOT = original_root
      run.set_verbose_output(original_verbose)
  ```

  This pattern works but is duplicated across multiple tests. No use of pytest fixtures or context managers to encapsulate this.

- **Suggestion**: Create a pytest fixture for verbose output state management:

  ```python
  @pytest.fixture
  def isolated_verbose_state():
      original = run.VERBOSE_OUTPUT
      yield
      run.set_verbose_output(original)
  ```

  Then use it in tests with `@pytest.mark.usefixtures("isolated_verbose_state")`. This centralizes cleanup logic and makes state isolation explicit.

### TST-4 · P3 · Missing edge case test for negative conditions in CLI parameter flow · 72%

- **File**: `src/tenzir_test/cli.py:136-141` and `src/tenzir_test/run.py:3369`
- **Issue**: No test verifies that `--verbose` and `--passthrough` flags interact correctly when both are provided
- **Reasoning**: The CLI accepts both `--verbose` and `--passthrough` flags independently. At line 3426 of run.py, the code uses OR logic: `set_verbose_output(verbose or passthrough_mode)`. There is no test covering the scenario where a user provides both flags explicitly, ensuring they don't conflict and that verbose output is enabled. This is a low-probability edge case but verifying it clarifies the intended interaction.
- **Evidence**:

  ```python
  # src/tenzir_test/run.py:3426
  set_verbose_output(verbose or passthrough_mode)
  ```

  No test calls `run_cli(..., verbose=True, passthrough=True, ...)` to verify the behavior.

- **Suggestion**: Add test `test_cli_both_verbose_and_passthrough_flags()` that verifies both flags can be provided together without conflict and that verbose output is correctly enabled.

### TST-5 · P3 · Incomplete state restoration in test for detailed summary · 70%

- **File**: `tests/test_run.py:947-968`
- **Issue**: Test `test_detailed_summary_order()` does not save/restore VERBOSE_OUTPUT state before calling `_print_detailed_summary()`
- **Reasoning**: The test calls `run._print_detailed_summary()` without setting or restoring the VERBOSE_OUTPUT global. This is technically not a problem if this function is called from a separate code path that doesn't depend on verbose state, but the implementation at line 3700 gates detailed summary printing on `is_verbose_output()`. If this test runs after another test that sets verbose=False, the detailed summary might not print. While `_print_detailed_summary()` itself doesn't check verbose state (it just prints unconditionally), the surrounding code in `run_cli()` does check before calling it. The test is isolated but relies on implicit test ordering.
- **Evidence**:

  ```python
  # tests/test_run.py:947-968
  def test_detailed_summary_order(capsys):
      summary = run.Summary(...)
      run._print_detailed_summary(summary)  # Direct call, no state setup
      run._print_ascii_summary(summary, ...)
  ```

  The function itself is designed to be called unconditionally, but the test doesn't clarify this.

- **Suggestion**: Either add a comment explaining that `_print_detailed_summary()` is called unconditionally by design, or add an explicit state setup to match how the integration calls it (with verbose check at caller level, not inside the function).

### TST-6 · P4 · Type annotation inconsistency in test function signature · 65%

- **File**: `tests/test_run.py:329`
- **Issue**: `test_success_includes_suite_suffix()` uses type annotations for `capsys` parameter but other similar tests don't
- **Reasoning**: At line 329-330, the function signature is `def test_success_includes_suite_suffix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:` with full type hints, while `test_handle_skip_uses_skip_glyph()` at line 309 uses no type hints. This is a minor style inconsistency that doesn't affect functionality but could indicate unintended difference in intent or test quality perception.
- **Evidence**:

  ```python
  # tests/test_run.py:329
  def test_success_includes_suite_suffix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:

  # tests/test_run.py:309
  def test_handle_skip_uses_skip_glyph(tmp_path, capsys):
  ```

- **Suggestion**: Standardize type hints across test functions. Either add full type hints to both or remove them for consistency with the rest of the test file's convention.
