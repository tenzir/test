#!/usr/bin/env python3
# runner: python
# fixtures: [http]
# timeout: 30

# Executor runs the Tenzir pipeline against the fixture-provided node.
executor = Executor()
pipeline = 'from {x: 42} | http f"{env("HTTP_FIXTURE_URL")}", body=this'
result = executor.run(pipeline)
print(result.stdout.decode(), end="")
