---
title: Example project tests updated for Tenzir v6
type: bugfix
authors:
  - mavam
prs:
  - 61
created: 2026-08-23T13:09:16.47158Z
---

The tests in `example-project` now run against Tenzir v6. They had drifted
into pipelines the current binary rejects, so the reference project could not
be used as a working example:

- `from_file` on an `.ndjson` file needs an explicit parser subpipeline
  (`{ read_ndjson }`).
- The `http` operator is gone. Requests that send each event as a body now use
  `each { from_http url, body=$this }`.
- Module-style names for built-in operators are deprecated, since modules are
  reserved for packages: `pipeline::detach`, `context::create_lookup_table`,
  `context::update`, and `context::inspect` become `pipeline_detach`,
  `context_create_lookup_table`, `context_update`, and `context_inspect`.

Baselines are unchanged, so the fixed pipelines produce the same output.
