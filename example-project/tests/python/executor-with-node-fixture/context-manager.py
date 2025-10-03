#!/usr/bin/env python3
# runner: python
# timeout: 30

# Context-manager style: `with` calls start() on enter and stop() on exit.
with acquire_fixture("node") as node:
    env = node.env
    # Get the Tenzir executor, with env adapted to work with the node fixture.
    executor = Executor.from_env(env)
    result = executor.run("remote { version } | summarize num_events=count()")
    print(result.stdout.decode(), end="")
