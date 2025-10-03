#!/usr/bin/env python3
# runner: python
# timeout: 30

# The Executor runs the `tenzir` binary with the configured environment.
executor = Executor()
pipeline = "from {xs: [1,2,3,4,5]} | unpack xs | summarize total=count()"
result = executor.run(pipeline)
if result.stdout:
    print(result.stdout.decode(), end="")
