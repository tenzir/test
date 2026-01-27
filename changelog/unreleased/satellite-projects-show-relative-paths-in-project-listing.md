---
title: Satellite projects show relative paths in project listing
type: change
authors:
  - mavam
  - claude
created: 2026-01-27T17:27:45.576827Z
---

When listing projects in the execution plan, satellite projects now display their path relative to the root project instead of just their directory name. This makes satellite projects with identical directory names distinguishable. Additionally, project markers have been refined: root projects use a filled marker (● for packages, ■ for regular projects), while satellite projects use an empty marker (○ for packages, □ for regular projects).
