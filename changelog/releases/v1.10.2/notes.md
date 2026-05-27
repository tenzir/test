This release lets pre-provisioned environments reuse installed inline Python dependencies and disable runtime installation entirely. It avoids unnecessary uv pip install calls for bare requirements that are already available.

## 🐞 Bug fixes

### Inline dependency installation control

Inline Python dependencies declared by tests or fixtures no longer force a runtime `uv pip install` when a bare dependency name is already available in the active Python environment.

Pass `--disable-inline-dependency-install`, or set `TENZIR_TEST_DISABLE_INLINE_DEPENDENCY_INSTALL=1`, to skip inline dependency installation entirely when another tool provisions the test environment. The harness still reads dependency metadata, but it leaves package installation to the caller.

*By @tobim and @codex in #49.*
