This patch preserves structured TQL diagnostics in test baselines while keeping regular logging output out of them.

## 🐞 Bug fixes

### Reliable TQL diagnostic baselines

TQL test baselines once again retain diagnostics before pipeline output, so warnings and errors remain covered by baseline comparisons.

*By @tobim in #59.*
