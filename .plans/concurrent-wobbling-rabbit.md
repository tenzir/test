# Plan: Make Test Output Verbosity Opt-in

## Summary

Add a `--verbose` / `-v` flag to make detailed per-test output opt-in. The new default (quiet mode) only shows failures and a compact summary, significantly reducing noise for large test suites.

## Behavior Changes

| Mode            | Passing tests | Skipped tests | Failed tests | Tree summary             |
| --------------- | ------------- | ------------- | ------------ | ------------------------ |
| Quiet (default) | Hidden        | Hidden        | Shown        | Hidden                   |
| Verbose (`-v`)  | Shown         | Shown         | Shown        | Shown (with `--summary`) |

## Implementation

### 1. Add global verbose state in `run.py` (~line 488)

```python
VERBOSE_OUTPUT = False

def set_verbose_output(enabled: bool) -> None:
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = enabled

def is_verbose_output() -> bool:
    return VERBOSE_OUTPUT
```

### 2. Modify `success()` function in `run.py` (line 2709)

Add early return when not verbose:

```python
def success(test: Path) -> None:
    if not is_verbose_output():
        return
    # ... rest unchanged
```

### 3. Modify `handle_skip()` function in `run.py` (line 3030)

Add early return for the print statement when not verbose:

```python
def handle_skip(reason: str, test: Path, update: bool, output_ext: str) -> bool | str:
    if is_verbose_output():
        rel_path = _relativize_path(test)
        suite_suffix = _format_suite_suffix()
        print(f"{SKIP} skipped {rel_path}{suite_suffix}: {reason}")
    # ... rest unchanged (ref file handling still runs)
```

### 4. Update summary logic in `run.py` (line 3683-3690)

Only show tree summary in verbose mode:

```python
_print_compact_summary(project_summary)
summary_enabled = show_summary or runner_summary or fixture_summary
if summary_enabled:
    if is_verbose_output():
        _print_detailed_summary(project_summary)  # Tree only in verbose
    _print_ascii_summary(
        project_summary,
        include_runner=runner_summary,
        include_fixture=fixture_summary,
    )
```

### 5. Add CLI flag in `cli.py` (after line 135)

```python
@click.option(
    "-v",
    "--verbose",
    "verbose",
    is_flag=True,
    help="Print individual test results as they complete.",
)
```

Update function signature (~line 165):

```python
def cli(..., verbose: bool, ...) -> int:
```

Pass to `run_cli()` (~line 180):

```python
result = runtime.run_cli(..., verbose=verbose, ...)
```

### 6. Update `run_cli()` in `run.py` (line 3336)

Add parameter:

```python
def run_cli(..., verbose: bool = False, ...) -> ExecutionResult:
```

Set mode after other settings (~line 3411):

```python
set_verbose_output(verbose or passthrough)
```

### 7. Update `execute()` in `run.py`

Add `verbose: bool = False` parameter for library API users.

## Files to Modify

- `src/tenzir_test/run.py` - Global state, success(), handle_skip(), summary logic, run_cli(), execute()
- `src/tenzir_test/cli.py` - Add --verbose/-v flag

## Verification

1. Run `tenzir-test` without flags - should only show failures and compact summary
2. Run `tenzir-test -v` - should show all individual results (current behavior)
3. Run `tenzir-test --summary` - should show compact summary + ASCII tables (no tree)
4. Run `tenzir-test -v --summary` - should show everything including tree
5. Run `tenzir-test -p` - passthrough mode should auto-enable verbose
