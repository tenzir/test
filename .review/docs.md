# Documentation Review

## Summary

The changes introduce a new `--verbose` / `-v` flag to make detailed test output opt-in. The implementation is complete and matches the plan documented in `.plans/concurrent-wobbling-rabbit.md`. However, several documentation gaps exist: the new flag is not documented in the public API docstrings, the behavior change is not captured in the changelog, and there is no update to external documentation.

## Findings

### DOC-1 · P2 · Missing docstring for verbose parameter · 88%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3373`
- **Issue**: The `verbose` parameter in `run_cli()` function lacks documentation in the docstring
- **Reasoning**: The function has a docstring "Execute the harness and return a structured result for library consumers." but does not document any parameters. The new `verbose` parameter (line 3369) changes behavior significantly by controlling whether individual test results are printed, but this is not explained for library consumers.
- **Evidence**:

  ```python
  def run_cli(
      # ... many parameters ...
      verbose: bool = False,
      # ... more parameters ...
  ) -> ExecutionResult:
      """Execute the harness and return a structured result for library consumers."""
  ```

  The docstring provides no parameter documentation at all. The `verbose` flag controls output behavior via `set_verbose_output(verbose or passthrough_mode)` at line 3426, affecting `success()`, `handle_skip()`, and summary printing.

- **Suggestion**: Expand the docstring to document all parameters, particularly `verbose` which now controls a key aspect of output behavior: "verbose: If True, print individual test results (pass/skip) as they complete. Automatically enabled in passthrough mode."

### DOC-2 · P2 · Missing docstring for verbose parameter in execute() · 88%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3803`
- **Issue**: The `verbose` parameter in `execute()` function lacks documentation in the docstring
- **Reasoning**: The `execute()` function is explicitly described as "Library-oriented wrapper around `run_cli` with defaulted parameters" but provides no parameter documentation. Since this is the primary public API for library consumers (as opposed to `run_cli`), the lack of parameter docs is more critical here.
- **Evidence**:

  ```python
  def execute(
      # ... many parameters ...
      verbose: bool = False,
      # ... more parameters ...
  ) -> ExecutionResult:
      """Library-oriented wrapper around `run_cli` with defaulted parameters."""
  ```

  The docstring mentions this is library-oriented but doesn't document what any parameter does, including the new `verbose` flag at line 3799.

- **Suggestion**: Add comprehensive parameter documentation to this public API function, documenting all parameters including: "verbose: Enable detailed per-test output. When False (default), only failures and summary are shown. When True, all test results including passes and skips are printed as they complete."

### DOC-3 · P3 · CLI help text doesn't explain quiet mode behavior · 82%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/cli.py:140`
- **Issue**: The help text "Print individual test results as they complete." doesn't explain what happens when the flag is not set
- **Reasoning**: The help text describes what `--verbose` does but doesn't clarify that the default (quiet mode) hides passing and skipped tests. Users may not understand the behavioral change from previous versions where all results were always shown.
- **Evidence**:

  ```python
  @click.option(
      "-v",
      "--verbose",
      is_flag=True,
      help="Print individual test results as they complete.",
  )
  ```

  This help text doesn't convey that the default behavior is now "quiet mode" where passes/skips are hidden. Compare to the plan which describes this as making output "opt-in."

- **Suggestion**: Update help text to: "Print individual test results as they complete. By default, only failures are shown." This makes the behavior change clearer to users.

### DOC-4 · P3 · Undocumented interaction between verbose and passthrough · 85%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:3426`
- **Issue**: The automatic enabling of verbose mode in passthrough mode is not documented
- **Reasoning**: At line 3426, the code sets `set_verbose_output(verbose or passthrough_mode)`, which means passthrough mode (`-p` flag) automatically enables verbose output. This interaction is not documented in either the CLI help text, the function docstrings, or comments.
- **Evidence**:

  ```python
  passthrough_mode = harness_mode is HarnessMode.PASSTHROUGH
  set_verbose_output(verbose or passthrough_mode)
  ```

  The passthrough flag's help text (cli.py:134) is "Stream raw test output directly to the terminal" which doesn't mention it also enables verbose mode for success/skip messages. The plan document mentions this at line 105 but it's not in user-facing docs.

- **Suggestion**: Add a comment explaining the logic: "# Passthrough mode requires verbose output to preserve test result messages" or update the passthrough help text to: "Stream raw test output directly to the terminal. Implies --verbose."

### DOC-5 · P3 · Missing changelog entry for new feature · 81%

- **File**: `/Users/mavam/code/tenzir/test/changelog/unreleased/`
- **Issue**: No changelog entry exists for the new `--verbose` flag and behavior change
- **Reasoning**: The changelog structure shows that user-facing changes should have entries in `changelog/unreleased/` (which gets moved to a release directory). This is a significant behavioral change - the default output is now quiet mode, which could surprise existing users. The project follows a structured changelog approach as evidenced by the release directories, but no entry exists for this feature.
- **Evidence**: The `changelog/unreleased/` directory is empty, but similar features in past releases have changelog entries (e.g., v0.12.0 has entries for CLI changes). This new feature changes default behavior significantly - previously all tests were shown, now only failures appear by default. The implementation plan in `.plans/concurrent-wobbling-rabbit.md` describes this as a major behavioral change.
- **Suggestion**: Add a changelog entry in `changelog/unreleased/` describing the new flag and behavior change: "Add `--verbose` / `-v` flag to opt into detailed test output. By default, only failures and summary are now shown, reducing noise for large test suites."

### DOC-6 · P4 · Global state functions lack docstrings · 80%

- **File**: `/Users/mavam/code/tenzir/test/src/tenzir_test/run.py:510-515`
- **Issue**: New functions `set_verbose_output()` and `is_verbose_output()` have no docstrings
- **Reasoning**: These are public module-level functions that control global state affecting output behavior. While their purpose may seem obvious from their names, they follow a pattern where similar functions like `set_harness_mode()` (line 517) also lack docstrings. However, documenting global state management is a documentation best practice, especially for functions that affect behavior in non-obvious ways across the codebase.
- **Evidence**:

  ```python
  def set_verbose_output(enabled: bool) -> None:
      global VERBOSE_OUTPUT
      VERBOSE_OUTPUT = enabled


  def is_verbose_output() -> bool:
      return VERBOSE_OUTPUT
  ```

  No docstrings present. These functions control whether `success()` and `handle_skip()` print output, but this relationship isn't documented.

- **Suggestion**: Add minimal docstrings: "Set whether to print individual test results (pass/skip) as they complete." and "Check if verbose output mode is enabled."
