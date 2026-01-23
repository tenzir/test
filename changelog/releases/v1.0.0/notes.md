This release introduces automatic binary detection with a clear precedence order: environment variables take priority, followed by PATH lookup, with `uvx` as fallback. Environment variables now support multi-part commands for full control over binary invocation. The `--tenzir-binary` and `--tenzir-node-binary` CLI flags have been removed in favor of environment variables.

## 💥 Breaking changes

### Auto-detect Tenzir binary with uvx fallback

Binary detection now follows a consistent precedence order: environment variable `TENZIR_BINARY` takes priority, followed by a PATH lookup, with `uvx` as the final fallback. The same logic applies to `TENZIR_NODE_BINARY`. Environment variables now support multi-part commands like `TENZIR_BINARY="uvx tenzir"`, giving you full control over how binaries are invoked. The `--tenzir-binary` and `--tenzir-node-binary` CLI flags have been removed in favor of environment variables, which provide more flexibility and clarity.

*By @mavam and @claude in #7.*
