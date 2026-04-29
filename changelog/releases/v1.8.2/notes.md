This release fixes suite-level requirement checks so they apply consistently across every test runner. Mixed TQL, shell, Python, and custom test suites now evaluate required Tenzir operators independently of runner order.

## 🐞 Bug fixes

### Runner-independent suite requirements

Suite-level `requires.operators` checks now apply consistently to every test runner. Mixed suites that combine TQL, shell, Python, or custom tests no longer depend on the first runner type to decide whether required Tenzir operators are available.

*By @mavam and @codex in #39.*
