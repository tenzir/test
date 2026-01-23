---
title: Auto-detect Tenzir binary with uvx fallback
type: breaking
authors:
  - mavam
  - claude
pr: 7
created: 2026-01-23T13:05:39.988216Z
---

Binary detection now follows a consistent precedence order: environment variable `TENZIR_BINARY` takes priority, followed by a PATH lookup, with `uvx` as the final fallback. The same logic applies to `TENZIR_NODE_BINARY`. Environment variables now support multi-part commands like `TENZIR_BINARY="uvx tenzir"`, giving you full control over how binaries are invoked. The `--tenzir-binary` and `--tenzir-node-binary` CLI flags have been removed in favor of environment variables, which provide more flexibility and clarity.
