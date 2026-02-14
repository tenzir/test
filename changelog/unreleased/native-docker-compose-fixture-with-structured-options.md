---
title: Native docker-compose fixture with structured options
type: feature
author: mavam
pr: 21
created: 2026-02-14T15:44:45.918487Z
---

Adds a built-in `docker-compose` fixture that starts and tears down Docker Compose services from structured fixture options.

The fixture validates Docker Compose availability, waits for service readiness (health-check first, running-state fallback), and exports deterministic service environment variables (including host-published ports).
