This release introduces a standalone fixture mode for starting fixtures without running tests, adds a built-in docker-compose fixture with structured options, and provides shared container runtime helpers for writing custom fixtures.

## 🚀 Features

### Native docker-compose fixture with structured options

Adds a built-in `docker-compose` fixture that starts and tears down Docker Compose services from structured fixture options.

The fixture validates Docker Compose availability, waits for service readiness (health-check first, running-state fallback), and exports deterministic service environment variables (including host-published ports).

*By @mavam in #21.*

### Shared container runtime helpers for fixtures

Writing container-based fixtures no longer requires duplicating boilerplate for runtime detection, process management, and readiness polling.

The new `container_runtime` module provides reusable building blocks that handle the common plumbing: detecting whether Docker or Podman is available, launching containers in detached mode, polling for service readiness, and tearing down cleanly. Custom fixtures can import these helpers and focus on their service-specific logic instead.

The built-in `docker-compose` fixture and the example project now use these shared helpers internally.

*By @mavam and @codex in #22.*

### Standalone fixture mode with --fixture CLI option

You can now start fixtures in foreground mode without running any tests by using the `--fixture` option. This lets you provision services (like a database or message broker) and use them in your workflow.

Specify fixture names or YAML-style configuration:

```sh
uvx tenzir-test --fixture mysql
uvx tenzir-test --fixture 'kafka: {port: 9092}' --debug
```

The harness prints fixture-provided environment variables like `HOST`, `PORT`, and `DATABASE`, then keeps services running until you press `Ctrl+C`. You can repeat `--fixture` to activate multiple fixtures. The option is mutually exclusive with positional TEST arguments.

*By @mavam and @claude in #20.*
