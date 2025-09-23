"""Command line entry point for the tenzir-test runner."""

from __future__ import annotations

from collections.abc import Sequence

from . import run as runtime


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the original runner while translating SystemExit to exit codes."""
    try:
        runtime.main(list(argv) if argv is not None else None)
    except SystemExit as exc:  # pragma: no cover - passthrough CLI termination
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
