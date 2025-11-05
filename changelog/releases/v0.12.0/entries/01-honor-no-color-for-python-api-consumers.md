---
title: Add configurable color output control
type: feature
authors:
  - codex
  - mavam
created: 2025-11-05
---

We default library consumers to plain output, keep the CLI smart about ANSI usage, and honour `NO_COLOR` across every helper while locking the behaviour down with tests.

When you `import tenzir_test.run`, the harness now emits unstyled text so CI logs stay machine-readable. You can still opt back into colours via `run.set_color_mode(run.ColorMode.ALWAYS)` or let auto-detection pick the best option.

Running `tenzir-test` in a terminal still shows coloured summaries, but redirecting stdout—or exporting `NO_COLOR=1`—forces the palette to plain text, covering diff hunks, failure trees, retry banners, and runner diagnostics.

A single palette helper now drives every glyph and symbol, and pytest coverage exercises CLI, library, and `NO_COLOR` modes to catch regressions immediately.
