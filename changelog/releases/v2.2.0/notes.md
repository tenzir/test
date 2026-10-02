You can now silence runtime warnings that are irrelevant to your Tenzir tests while keeping errors and compilation diagnostics visible. This release also updates the example project for Tenzir v6 and removes the deprecated `--multi` option.

## 🚀 Features

### Runtime warning suppression for Tenzir tests

You can now suppress runtime warnings from selected Tenzir tests with `quiet: true` in their frontmatter:

```tql
---
quiet: true
---
from {time: "not a timestamp"}
time = time.parse_time("%Y-%m-%d")
```

The setting follows the TQL `quiet` operator: events and bytes are unchanged, errors still fail the test, and compilation diagnostics remain visible. It works in comparison, update, and passthrough modes and requires a Tenzir binary that supports `quiet`.

Set `quiet: true` in `test.yaml` to apply it to a directory and its descendants. Individual tests or child directories can restore runtime warnings with `quiet: false`. Warning suppression is disabled by default.

*By @mavam in #63.*

## 🔧 Changes

### Removal of the deprecated multi-executor option

Test runs no longer pass the deprecated `--multi` Tenzir option.

*By @aljazerzen in #62.*

## 🐞 Bug fixes

### Example project tests updated for Tenzir v6

The tests in `example-project` now run against Tenzir v6. They had drifted into pipelines the current binary rejects, so the reference project could not be used as a working example:

- `from_file` on an `.ndjson` file needs an explicit parser subpipeline (`{ read_ndjson }`).
- The `http` operator is gone. Requests that send each event as a body now use `each { from_http url, body=$this }`.
- Module-style names for built-in operators are deprecated, since modules are reserved for packages: `pipeline::detach`, `context::create_lookup_table`, `context::update`, and `context::inspect` become `pipeline_detach`, `context_create_lookup_table`, `context_update`, and `context_inspect`.

Baselines are unchanged, so the fixed pipelines produce the same output.

*By @mavam in #61.*
