---
title: Add pre-compare transforms for non-deterministic output
type: feature
authors:
  - mavam
  - claude
created: 2026-01-30T20:46:00.000000Z
---

The test framework now supports pre-compare transforms that normalize output before comparison with baselines. This helps handle tests with non-deterministic output like unordered result sets from hash-based aggregations or parallel operations.

Configure the `pre-compare` option in `test.yaml` or per-test frontmatter to apply transforms to both actual output and baselines before comparison:

```yaml
# Sort output lines for comparison (baseline stays unchanged)
pre-compare: sort
```

The `sort` transform sorts output lines lexicographically, making it easy to handle unordered results. Transforms only affect comparison—baseline files remain untransformed on disk, and `--update` continues to store original output.
