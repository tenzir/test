---
title: Shell test files default to shell runner
type: bugfix
authors:
  - mavam
  - codex
pr: 30
created: 2026-02-24T16:32:14.824321Z
---

Shell test files (`.sh`) now always default to the "shell" runner, even when a directory-level `test.yaml` file specifies a different runner (for example, `runner: tenzir`). This makes shell scripts work reliably in mixed-runner directories without requiring explicit `runner:` frontmatter in each file. Explicit `runner:` declarations in test file frontmatter still take precedence and can override this behavior if needed.
