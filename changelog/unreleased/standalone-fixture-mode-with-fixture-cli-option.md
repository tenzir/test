---
title: Standalone fixture mode with --fixture CLI option
type: feature
authors:
  - mavam
  - claude
pr: 20
created: 2026-02-14T15:30:18.036656Z
---

You can now start fixtures in foreground mode without running any tests by using the `--fixture` option. This lets you provision services (like a database or message broker) and use them in your workflow.

Specify fixture names or YAML-style configuration:

```sh
uvx tenzir-test --fixture mysql
uvx tenzir-test --fixture 'kafka: {port: 9092}' --debug
```

The harness prints fixture-provided environment variables like `HOST`, `PORT`, and `DATABASE`, then keeps services running until you press `Ctrl+C`. You can repeat `--fixture` to activate multiple fixtures. The option is mutually exclusive with positional TEST arguments.
