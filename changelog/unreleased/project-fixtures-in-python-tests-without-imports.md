---
title: Project fixtures in Python tests without imports
type: feature
authors:
  - mavam
prs:
  - 66
created: 2026-11-13T17:49:04.623557Z
---

Python tests now find project fixtures without importing them. Before, the process that runs a Python test knew only the built-in fixtures, so `acquire_fixture` failed for a project fixture unless the test imported the project's `fixtures` package first, and a test in a satellite project could not reach the root project's fixtures at all.

The harness now hands its project roots to that process, which registers the same fixtures as the harness: those of the root project and, in a satellite, the satellite's own. A test can then acquire any of them directly:

```python
# No `import fixtures` needed for a fixture that the project defines.
with acquire_fixture("http") as http:
    print(http.env["HTTP_FIXTURE_URL"])
```

An explicit `import fixtures` keeps working and still yields the test's own project fixtures. Importing a fixture module a second time no longer fails with "already registered", because registering the same fixture definition again replaces it.
