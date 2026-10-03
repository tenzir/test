You can now export final test outcomes, failure diagnostics, and plain diffs as versioned JSON reports. Tagged log records let your CI surface failures from sandboxed builds without parsing terminal output.

## 🚀 Features

### Structured test reports for CI

You can now export final test outcomes, failure diagnostics, and plain diffs as a versioned JSON report:

```sh
tenzir-test --root test --report-json artifacts/tests.json
```

Use `--report-json -` to forward tagged JSON records through sandboxed build logs. Reports retain partial results and final retry outcomes so CI can show failures without parsing terminal output.

*By @mavam in #64.*
