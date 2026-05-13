---
title: Project fixture inline dependencies
type: bugfix
authors:
  - mavam
  - codex
pr: 47
created: 2026-05-13T09:10:22.248071Z
---

Project fixtures can now declare Python package dependencies inline with PEP
723 metadata:

```python
# /// script
# dependencies = ["boto3"]
# ///
```

`tenzir-test` installs these dependencies with `uv` before importing fixture
modules, so fixtures used during regular test runs and `tenzir-test --fixture`
can manage their own Python package requirements.
