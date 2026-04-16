This release fixes misleading aggregate test summary percentages in the CLI. Runs with a small number of failures now report non-perfect pass and fail rates, so mixed outcomes no longer appear as `100%` passed and `0%` failed.

## 🐞 Bug fixes

### Accurate aggregate pass and fail percentages

The final aggregate summary now reports non-perfect pass and fail percentages whenever at least one executed test fails. Previously, rounding could show `100%` passed and `0%` failed for runs with a small number of failures, even though the overall result was not a full success.

For example, a run like `586 passed / 1 failed / 152 skipped` now renders the executed-test percentages as `99%` passed and `1%` failed. This makes mixed outcomes easier to spot at a glance in the CLI output.

*By @mavam and @codex.*
