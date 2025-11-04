Expose tenzir-test as a reusable Python library.

## 🚀 Features

### Expose tenzir-test as a library

The test framework now exposes its harness as a reusable Python library, returning structured execution results instead of exiting the process. Callers can import the new `execute` helper and receive an `ExecutionResult` with project summaries, exit codes, and failure details for downstream automation.

Example:

```python
from pathlib import Path

from tenzir_test import ExecutionResult, execute

result: ExecutionResult = execute(tests=[Path("tests/pipeline.tql")])
if result.exit_code:
    raise SystemExit(result.exit_code)
for project in result.project_results:
    print(project.selection.root, project.summary.total)
```

*By @mavam and @codex.*
