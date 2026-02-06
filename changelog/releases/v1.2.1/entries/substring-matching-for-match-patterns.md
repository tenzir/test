---
title: Simpler substring matching for -m/--match flag
type: change
authors:
  - mavam
  - claude
pr: 16
created: 2026-02-06T18:55:06.331758Z
---

The `-m`/`--match` flag now treats bare strings as substring matches automatically.

Previously, you had to use glob syntax like `tenzir-test -m '*mysql*'` to match tests by name. Now you can write `tenzir-test -m mysql` and it automatically matches any test path containing "mysql". This makes the common case of substring matching simpler and more intuitive.

Patterns that contain glob metacharacters (`*`, `?`, `[`) still use fnmatch syntax as before, so existing patterns with wildcards continue to work exactly as they did. This change is fully backwards-compatible.
