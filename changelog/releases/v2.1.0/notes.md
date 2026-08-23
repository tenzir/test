Tenzir Test now keeps multiline record blocks intact when sorting test output, making nondeterministic event order easier to stabilize. The shell, Python fixture, and diff runners also apply pre-compare transforms consistently before comparing baselines.

## 🔧 Changes

### Format-aware pre-compare sort

The `pre-compare: sort` transform now keeps multiline record blocks intact instead of sorting individual lines. Sorting a TQL test's default output no longer scrambles the fields of each event. A block runs from a `{` alone on a line in the first column through the matching `}` in the first column.

This makes it practical to stabilize non-deterministic event order without putting test scaffolding into the pipeline under test:

```yaml
---
pre-compare: sort
---
```

Tests that end in a trailing `sort <field>` operator solely to make output deterministic can drop that operator and declare `pre-compare: sort` in frontmatter or `test.yaml` instead.

NDJSON stays one event per line and sorts as before. Output without first-column braces also sorts line by line, unchanged. Two shapes are still not grouped and remain unsuitable for `pre-compare: sort`: indented structured output such as a pretty-printed JSON array, and multiline diagnostics, whose lines sort independently of their header.

The shell, Python fixture, and diff runners now also apply `pre-compare` transforms before comparing against the baseline; previously they accepted the option and ignored it. Baselines written with `--update` still record raw, untransformed output.

*By @mavam in #60.*
