---
title: Python 3.13 minimum requirement
type: change
authors:
  - mavam
  - codex
prs:
  - 44
created: 2026-05-12T13:53:23.650011Z
---

`tenzir-test` now requires Python 3.13 or newer.

Users on Python 3.12 need to upgrade their interpreter before installing or running the CLI:

```sh
uvx --python 3.13 tenzir-test --help
```
