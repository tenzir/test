tenzir-test now lets users select scenarios by requested fixture name, making it easier to run only tests that depend on resources such as nodes or Docker Compose. This release also requires Python 3.13 or newer.

## 🚀 Features

### Fixture name test selection

Select tests by requested fixture name with the new `--fixture-name` option:

```sh
tenzir-test --fixture-name node
tenzir-test tests/alerts --match kafka --fixture-name docker-compose
```

`--fixture-name` can be repeated and combines with `--fixture-tag` using OR semantics before intersecting with positional test paths and `--match`. Fixture selectors are long-only; the previous `-F` alias for `--fixture-tag` has been removed before the CLI shape settles.

*By @mavam and @codex.*

## 🔧 Changes

### Python 3.13 minimum requirement

`tenzir-test` now requires Python 3.13 or newer.

Users on Python 3.12 need to upgrade their interpreter before installing or running the CLI:

```sh
uvx --python 3.13 tenzir-test --help
```

*By @mavam and @codex in #44.*
