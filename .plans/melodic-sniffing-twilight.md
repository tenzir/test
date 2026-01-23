# Plan: Address Review Findings for --verbose Flag

## Summary

Address 14 findings from code review of the `--verbose` flag implementation. The changelog already exists and is comprehensive, so DOC-5 is already resolved.

## Changes

### 1. Documentation: Improve help text (DOC-3, DOC-4, ARC-1)

**File:** `src/tenzir_test/cli.py`

Update `--verbose` help text to explain default behavior and passthrough coupling:

```python
# Line 140
help="Print individual test results as they complete. By default, only failures are shown. Automatically enabled in passthrough mode."
```

### 2. Documentation: Add docstrings (DOC-1, DOC-2, DOC-6)

**File:** `src/tenzir_test/run.py`

Add docstrings to:

- `set_verbose_output()` (line 510)
- `is_verbose_output()` (line 515)
- Document `verbose` parameter in `run_cli()` (line 3373) and `execute()` (line 3803)

### 3. Readability: Add comment explaining success/fail asymmetry (RDY-4)

**File:** `src/tenzir_test/run.py`

Add comment before `fail()` function (line 2729) explaining that failures always print regardless of verbose mode because failures are critical information.

### 4. Readability: Add comment for passthrough coupling (RDY-3)

**File:** `src/tenzir_test/run.py`

Add clarifying comment at line 3426:

```python
# Passthrough mode requires verbose output to show real-time test results
set_verbose_output(verbose or passthrough_mode)
```

### 5. Tests: Use getter instead of direct access (RDY-2)

**File:** `tests/test_run.py`

Change lines 311 and 331 from:

```python
original_verbose = run.VERBOSE_OUTPUT
```

to:

```python
original_verbose = run.is_verbose_output()
```

### 6. Tests: Add missing test cases (TST-1, TST-2)

**File:** `tests/test_run.py`

Add two new tests after `test_success_includes_suite_suffix`:

1. `test_handle_skip_suppressed_when_verbose_disabled()` - verify no output when verbose=False
2. `test_success_suppressed_when_verbose_disabled()` - verify no output when verbose=False

## Skipped Findings

| Finding | Reason                                                                                                         |
| ------- | -------------------------------------------------------------------------------------------------------------- |
| DOC-5   | Changelog already exists at `changelog/unreleased/verbose-output-flag-for-test-results.md`                     |
| UXD-3   | Startup feedback for verbose mode is unnecessary noise; the behavior is self-evident from output               |
| TST-3   | Creating a pytest fixture for VERBOSE_OUTPUT isolation is scope creep; current try/finally pattern is adequate |
| TST-4   | Testing both `--verbose` and `--passthrough` together is unnecessary given the simple OR logic                 |
| RDY-1   | Renaming `--verbose` to `--show-progress` would break expectations; `--verbose` is CLI convention              |
| RDY-5   | Variable naming `original_verbose` is clear enough in context                                                  |
| UXD-2   | This is intentional behavior documented in the changelog                                                       |
| UXD-4/5 | Minor UX concerns that don't warrant code changes                                                              |

## Files to Modify

1. `src/tenzir_test/cli.py` - help text update
2. `src/tenzir_test/run.py` - docstrings and comments
3. `tests/test_run.py` - fix encapsulation, add tests

## Verification

1. Run `uv run ruff check src/tenzir_test/cli.py src/tenzir_test/run.py tests/test_run.py`
2. Run `uv run mypy src/tenzir_test/cli.py src/tenzir_test/run.py tests/test_run.py`
3. Run `uv run pytest tests/test_run.py -v -k "verbose or skip or success"`
