# Readability Review

## Summary

The changes add a new `--verbose` flag to control per-test output verbosity through a global `VERBOSE_OUTPUT` state variable and setter/getter functions. The implementation follows the existing pattern for similar configuration options (`SHOW_DIFF_OUTPUT`, `SHOW_DIFF_STAT`) and integrates cleanly into the CLI. However, the semantic relationship between the verbose flag and output functions exhibits some naming ambiguity and potential inconsistency in behavior expectations.

## Findings

### RDY-1 · P3 · Inconsistent naming between flag intent and output functions · 78%

- **File**: `src/tenzir_test/cli.py:137-141`, `src/tenzir_test/run.py:510-516`, `src/tenzir_test/run.py:2719-2726`, `src/tenzir_test/run.py:3042-3046`
- **Issue**: The `--verbose` flag name suggests it enables verbose logging or additional detail, but its implementation only conditionally suppresses success messages and skip messages. This creates ambiguity about what "verbose" means in this context.
- **Reasoning**: The flag's help text "Print individual test results as they complete" is clear, but the internal naming creates conceptual mismatch:
  - `is_verbose_output()` returns true when individual results should be printed
  - `success()` prints nothing unless `is_verbose_output()` is true
  - `handle_skip()` prints nothing unless `is_verbose_output()` is true

  This means "verbose" actually means "show individual results" rather than "provide more information." The naming works but could confuse readers unfamiliar with the codebase who might expect verbose mode to add detail to existing output, not conditionally enable whole output functions.

- **Evidence**:
  - At line 140, help text: "Print individual test results as they complete"
  - At line 2720-2721: `if not is_verbose_output(): return` gates the entire success message
  - At line 3043-3046: skip message only prints when `is_verbose_output()` is true
  - Line 3426: `set_verbose_output(verbose or passthrough_mode)` shows verbose mode is also automatically enabled in passthrough mode, further obscuring the primary intent

- **Suggestion**: Consider renaming to clarify intent. Options:
  1. Rename flag to `--show-progress` or `--show-results` to better reflect that it shows individual test outcomes as they complete
  2. Add a clarifying comment explaining that "verbose" in this context means "show per-test completion messages"
  3. Consider whether `fail()` should also be gated by this flag for consistency (currently it always prints)

### RDY-2 · P3 · Test setup pattern breaks encapsulation of global state · 75%

- **File**: `tests/test_run.py:310-326`, `tests/test_run.py:330-344`
- **Issue**: Tests directly access and restore global module state (`run.VERBOSE_OUTPUT`) rather than using the provided setter functions throughout the restoration process. This couples tests to internal implementation details.
- **Reasoning**: The tests correctly use `set_verbose_output()` to set the state but then directly assign to `run.VERBOSE_OUTPUT` during cleanup. This pattern:
  - Requires knowledge of the module's internal variable name
  - Bypasses the setter function, breaking the abstraction
  - Makes tests fragile if the implementation changes (e.g., if logging were involved)
  - Inconsistent with how other globals are restored (e.g., `run.ROOT`)

- **Evidence**:

  ```python
  # test_handle_skip_uses_skip_glyph (line 310-326)
  original_verbose = run.VERBOSE_OUTPUT  # Direct access
  try:
      run.set_verbose_output(True)  # Using setter
      # ...
  finally:
      run.set_verbose_output(original_verbose)  # Could use setter
  ```

  The test saves and restores the global directly but uses the setter to change it. Line 326 and line 344 could use the setter for consistency.

- **Suggestion**: Use `set_verbose_output()` for both setting and restoring:

  ```python
  original_verbose = run.is_verbose_output()  # Get current state
  try:
      run.set_verbose_output(True)
      # test code
  finally:
      run.set_verbose_output(original_verbose)
  ```

  This maintains abstraction and is more resilient to implementation changes. If the internal representation becomes more complex (e.g., nested context), only the setter needs updating.

### RDY-3 · P3 · Behavioral coupling between verbose flag and passthrough mode is implicit · 72%

- **File**: `src/tenzir_test/run.py:3426`
- **Issue**: The line `set_verbose_output(verbose or passthrough_mode)` silently enables verbose output in passthrough mode, but this coupling is not immediately obvious from reading either the flag setup or passthrough mode logic.
- **Reasoning**: A reader encountering this line without context might not understand why passthrough mode automatically enables verbose output. This is a sensible design choice (passthrough likely needs to show real-time output) but the connection isn't explained. Future maintainers might accidentally break this assumption when refactoring.

- **Evidence**:
  - Line 3426: `set_verbose_output(verbose or passthrough_mode)`
  - No comment explaining the relationship between passthrough mode and verbose output
  - Help text for `--verbose` flag (line 140) doesn't mention it's automatically enabled in passthrough mode
  - Help text for `--passthrough` flag (line 134) doesn't mention it implies verbose output

- **Suggestion**: Add a clarifying comment explaining the coupling:

  ```python
  # In passthrough mode, enable verbose output to show real-time test results
  set_verbose_output(verbose or passthrough_mode)
  ```

  Optionally, update the help text for `--verbose` to mention this: "Print individual test results as they complete (automatically enabled in passthrough mode)."

### RDY-4 · P2 · Conditional logic inconsistency in success/fail functions · 82%

- **File**: `src/tenzir_test/run.py:2719-2735`
- **Issue**: The `success()` function checks `is_verbose_output()` and returns early if false, while `fail()` has no such check and always prints. This asymmetry could confuse readers and may indicate incomplete implementation.
- **Reasoning**: The functions appear to be a matching pair for test outcome reporting, but they have inconsistent behavior:
  - `success()` (line 2719-2726): Guards output with `if not is_verbose_output(): return`
  - `fail()` (line 2729-2734): No guard; always prints

  This asymmetry suggests either:
  1. `fail()` should also be guarded (intended incomplete refactor), or
  2. `fail()` intentionally always prints because failures are critical (needs documentation)

- **Evidence**:

  ```python
  def success(test: Path) -> None:
      if not is_verbose_output():
          return
      with stdout_lock:
          # print success

  def fail(test: Path) -> None:
      with stdout_lock:
          # print failure (no guard)
  ```

- **Suggestion**: Clarify intent with one of these approaches:
  1. Add guard to `fail()`: `if not is_verbose_output(): return` (if failures should also be suppressed in non-verbose mode)
  2. Add comment to `fail()` explaining why it always prints despite `success()` being conditional
  3. Consider whether test failures should be visible regardless of verbose mode (they probably should be)

### RDY-5 · P3 · Global variable naming uses negative semantics in test files · 70%

- **File**: `tests/test_run.py:311, 331`
- **Issue**: The local variables `original_verbose` store the saved state, but readers must understand that this value will be used to restore a boolean flag. The name doesn't clarify what "verbose" refers to (e.g., is it a feature name, output mode, or something else?).
- **Reasoning**: While technically correct, local variable names like `original_verbose` lack context. In a test file with many similar patterns (saving/restoring globals like `original_root`), the meaning is clear from convention. However, the variable name doesn't indicate:
  - That it's a boolean state value
  - What it controls (verbose output mode specifically)
  - That it will be restored to turn off the feature

  More explicit naming would improve scannability: `original_verbose_output_state` or `original_show_results` would be immediately clear.

- **Evidence**:

  ```python
  original_verbose = run.VERBOSE_OUTPUT  # What exactly is "verbose" controlling?
  # ... later ...
  run.set_verbose_output(original_verbose)
  ```

  A reader skimming the test needs to trace back to understand the semantic meaning.

- **Suggestion**: Use more explicit variable names:

  ```python
  original_verbose_output = run.is_verbose_output()
  try:
      run.set_verbose_output(True)
      # test
  finally:
      run.set_verbose_output(original_verbose_output)
  ```

  This immediately clarifies that the variable holds output verbosity state. Alternatively: `original_show_individual_results` if the concept warrants a more specific name.
