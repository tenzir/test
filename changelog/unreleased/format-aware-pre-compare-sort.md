---
title: Format-aware pre-compare sort
type: change
authors:
  - mavam
prs:
  - 60
created: 2026-08-23T12:14:06.588479Z
---

The `pre-compare: sort` transform now keeps multiline record blocks intact
instead of sorting individual lines. Sorting a TQL test's default output no
longer scrambles the fields of each event.

This makes it practical to stabilize non-deterministic event order without
putting test scaffolding into the pipeline under test:

```yaml
---
pre-compare: sort
---
```

Tests that end in a trailing `sort <field>` operator solely to make output
deterministic can drop that operator and declare `pre-compare: sort` in
frontmatter or `test.yaml` instead. Plain text and NDJSON output still sort line
by line, so existing baselines remain valid.

The shell and Python fixture runners now also apply `pre-compare` transforms
before comparing against the baseline; previously they accepted the option and
ignored it.
