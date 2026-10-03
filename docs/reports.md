# Structured test reports

Use `--report-json` to collect test results without parsing terminal output:

```sh
tenzir-test --root test --report-json artifacts/tests.json --report-root .
```

Reports do not change the test exit code or terminal output. They work with
`--no-hooks`, `--no-diff`, parallel jobs, suites, retries, and satellite projects.
`--report-root` controls the base directory for paths and defaults to the current
working directory. Paths use forward slashes and can contain `..` for projects
outside that directory. Consumers must validate paths before linking or
annotating source files.

## File format

The report is a UTF-8 JSON document with these fields:

- `schema_version`: The format version, currently `1`.
- `root`: The absolute base directory used to resolve paths.
- `exit_code`: The harness exit code.
- `interrupted`: Whether the run was interrupted.
- `errors`: Harness-level errors, such as configuration or hook failures.
- `summary`: Counts named `total`, `passed`, `failed`, and `skipped`.
- `tests`: Final results sorted by project and path.

Each test result contains:

- `path` and `project`: The test file and project directory, relative to `root`.
- `runner` and `suite`: The runner name and optional suite name.
- `outcome`: `passed`, `failed`, or `skipped`.
- `attempts` and `duration`: The attempt count and elapsed seconds.
- `reason`: A failure diagnostic or skip reason, when available.
- `stdout`, `stderr`, and `returncode`: Captured subprocess output and exit code
  for a failing test. These describe the last subprocess captured through the
  harness's `run_subprocess` helper. Output from custom runners that bypass that
  helper or stream output in passthrough mode is not captured.
- `diff`: Plain unified diffs, including reference filenames, for mismatched
  expectations. Multiple comparisons are concatenated.
- `truncated`: Whether any diagnostic was shortened.

Diagnostic text has no ANSI styling. Each text field is bounded to 6,000
characters after JSON encoding. Keep raw logs as a companion artifact when you
need complete output. Only the final retry contributes diagnostics; a test that
passes after retrying has no failure output.

A suite-level fixture failure appears as a failed result for the suite's
`test.yaml`, with `runner` set to `suite`. It contributes to report counts even
if no individual test ran. Harness-level errors do not increase failed-test
counts, so consumers must also check `exit_code`, `errors`, and `interrupted`.

The harness writes reports on normal completion, test failures, harness errors,
and handled interruptions. It atomically replaces an existing report and creates
parent directories when needed. A forced process kill cannot write a final
report. An error writing a report fails the invocation rather than silently
leaving an old result behind.

The library API accepts `report_json=Path(...)` and `report_root=Path(...)` on
`tenzir_test.run.execute` and `run_cli`.

## Forward reports through build logs

When tests run in a build sandbox, use `--report-json -`:

```sh
tenzir-test --root test --report-json - --report-root .
```

Instead of a JSON document, this emits tagged JSON lines alongside normal
terminal output. Every record begins with the literal `TENZIR_TEST_REPORT `,
followed by a JSON object containing `schema_version: 1` and an `event`:

- `start`: Contains `root` and begins one invocation.
- `test`: Contains one final test result in `test`.
- `finish`: Contains `exit_code`, `interrupted`, `errors`, and `summary`.

Records are flushed as tests complete, so consumers can preserve partial results
when the build is interrupted. A missing `finish` means the invocation is
incomplete, not successful. Multiple invocations can share one build log; each
`start` begins a separate report. Build tools may prepend a log prefix to each
line. Consumers should remove that prefix, decode only tagged records, and keep
the remaining lines as human-readable logs. Do not infer failures from terminal
symbols or parse diffs from display formatting.

Within schema version 1, consumers should ignore unknown fields and events.
Incompatible format changes increment `schema_version`.
