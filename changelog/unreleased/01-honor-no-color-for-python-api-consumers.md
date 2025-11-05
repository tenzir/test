---
title: Add configurable color output control
type: feature
authors:
  - codex
  - mavam
created: 2025-11-05
---

We switched library consumers to plain output by default and added full `NO_COLOR` support, keeping ANSI colors only where they enhance the interactive experience.

When you `import tenzir_test.run` directly, the harness now emits unstyled text by default—perfect for log parsers, CI systems, and automation scripts that expect machine-readable output. If you need colors back, call `run.set_color_mode(run.ColorMode.ALWAYS)` to force ANSI sequences on, or use `run.ColorMode.AUTO` to let the harness detect terminal capabilities.

The `tenzir-test` CLI continues to auto-detect your terminal and renders colors when you run tests interactively. Redirect stdout to a file or pipe, and it automatically switches to plain mode. Set `NO_COLOR=1` in your environment, and every output—failure trees, diff hunks, retry banners—strips its color codes.

All palette logic now lives in a single helper that colorizes checkmarks, crosses, diff lines, and progress indicators consistently. This unification means `run.print_diff(...)` respects your color mode choice everywhere, whether you invoke it from the CLI or from library code.

We added pytest coverage for all three modes—CLI with colors, library without colors, and `NO_COLOR=1` forcing plain output—so future changes won't break the contract.
