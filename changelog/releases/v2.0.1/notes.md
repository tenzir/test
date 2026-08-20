This patch preserves compiler diagnostics in diff-runner baselines, so expected errors continue to be validated consistently with pipeline output.

## 🐞 Bug fixes

### Reliable compiler diagnostic baselines

Diff-based tests once again retain compiler diagnostics in their baselines, so expected errors are validated consistently with normal pipeline output.

*By @tobim in #57.*
