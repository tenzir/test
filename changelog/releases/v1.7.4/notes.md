This release fixes how skipped tests interact with baseline files. Existing baselines now stay untouched when you skip tests, which prevents unrelated baseline churn and avoids false failures when you toggle skip conditions.

## 🐞 Bug fixes

### Skipped tests preserve existing baseline files

Skipping a test no longer modifies its baseline file. Previously, running with `--update` would overwrite the baseline of a skipped test with an empty file, and running without `--update` would fail if the baseline was non-empty. Skipped tests now leave existing baselines untouched, so toggling a skip condition no longer causes unrelated baseline churn.

*By @mavam and @codex.*
