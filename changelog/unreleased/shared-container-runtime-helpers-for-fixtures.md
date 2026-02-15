---
title: Shared container runtime helpers for fixtures
type: feature
authors:
  - mavam
  - codex
pr: 22
created: 2026-02-15T09:54:01.330622Z
---

Writing container-based fixtures no longer requires duplicating boilerplate for
runtime detection, process management, and readiness polling.

The new `container_runtime` module provides reusable building blocks that handle
the common plumbing: detecting whether Docker or Podman is available, launching
containers in detached mode, polling for service readiness, and tearing down
cleanly. Custom fixtures can import these helpers and focus on their
service-specific logic instead.

The built-in `docker-compose` fixture and the example project now use these
shared helpers internally.
