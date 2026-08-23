#!/usr/bin/env python
"""Exercise the HTTP fixture via the Executor helper."""

tenzir = Executor()
tql = 'from {x: 42} | each { from_http env("HTTP_FIXTURE_URL"), body=$this }'
result = tenzir.run(tql)
print(result.stdout.decode(), end="")
