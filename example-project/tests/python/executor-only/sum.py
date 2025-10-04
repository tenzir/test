#!/usr/bin/env python3
# runner: python
# timeout: 10

# The Executor runs the `tenzir` binary with the configured environment.
result = Executor().run("from {xs: [1,2,3,4,5]} | sum=xs.sum()")
print(result.stdout.decode(), end="")
