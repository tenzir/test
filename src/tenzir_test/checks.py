from __future__ import annotations

import shlex
import subprocess
from typing import Sequence

_COMMANDS: Sequence[Sequence[str]] = (
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("mypy",),
    ("pytest",),
    ("uv", "build"),
)


def _run(command: Sequence[str]) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"> {printable}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    for command in _COMMANDS:
        _run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
