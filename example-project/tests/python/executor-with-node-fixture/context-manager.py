#!/usr/bin/env python
# timeout: 10

# Context-manager style: `with` calls start() on enter and stop() on exit.
with acquire_fixture("node") as node:
    # Perform a side-effect on the persistent state of the node.
    Executor.from_env(node.env).run("from {x: 42} | import")

# Now we restart the node and re-use the persistent state.
with acquire_fixture("node") as node:
    # Regurgitate the imported value of x.
    result = Executor.from_env(node.env).run("export")
    print(result.stdout.decode(), end="")
