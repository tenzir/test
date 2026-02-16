This release adds fine-grained controls for running skipped tests, including a new --run-skipped-reason flag with substring and glob matching semantics.

## 🚀 Features

### Add fine-grained run-skipped selectors

`tenzir-test` now supports both coarse and fine-grained controls for running skipped tests.

Use `--run-skipped` as a sledgehammer, or `--run-skipped-reason` with the same matching semantics as `--match` (bare substring or glob):

```sh
tenzir-test --run-skipped
tenzir-test --run-skipped-reason 'maintenance'
tenzir-test --run-skipped-reason '*docker*'
```

If both options are provided, `--run-skipped` takes precedence.

*By @mavam and @codex in #23 and #24.*
