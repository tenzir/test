#!/usr/bin/env python
# fixtures: [http]
# timeout: 30

# Executor runs the Tenzir pipeline against the fixture-provided node.
tenzir = Executor()
tql = 'from {x: 42} | http f"{env("HTTP_FIXTURE_URL")}", body=this'
result = tenzir.run(tql)
print(result.stdout.decode(), end="")
