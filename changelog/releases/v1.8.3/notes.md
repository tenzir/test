This release makes hook debug diagnostics consistent with the rest of the harness debug trace, so users get uniform output when diagnosing hook behavior.

## 🐞 Bug fixes

### Consistent hook debug diagnostics

Hook diagnostics emitted with `--debug` now use the same formatting as the rest of the harness debug trace. Previously, hook invocation messages used ad-hoc `debug:` lines, which made debug output inconsistent.

*By @mavam and @codex in #40.*
