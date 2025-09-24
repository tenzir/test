#!/usr/bin/env python3
# runner: python
# timeout: 30
from __future__ import annotations

import sys

from tenzir_test.fixtures import Executor, requested


def main() -> int:
    fixtures = requested()
    if fixtures.has("sink"):
        print("# using sink fixture\n", end="")
    executor = Executor()
    query = (
        "\n".join(
            [
                'from_file f"{env("TENZIR_INPUTS")}/events.ndjson"',
                "where severity >= 5",
                "summarize count=count()",
            ]
        )
        + "\n"
    )
    result = executor.run(query)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.decode(), end="")
        if result.stderr:
            print(result.stderr.decode(), end="", file=sys.stderr)
        return result.returncode
    print(result.stdout.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
