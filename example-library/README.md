# Example Library

This directory contains two sibling packages, `foo` and `bar`, to demonstrate
explicit package loading. Point `tenzir-test` at the directory and add
`--package-dirs example-library` so both packages are visible to every test:

```sh
uvx tenzir-test --package-dirs example-library example-library
```

## Layout

```
example-library/
├── foo/
│   ├── package.yaml
│   ├── test.yaml
│   ├── operators/
│   │   └── increment.tql
│   └── tests/
│       ├── use-bar.tql
│       └── use-bar.txt
└── bar/
    ├── package.yaml
    ├── test.yaml
    ├── operators/
    │   └── double.tql
    └── tests/
        ├── use-foo.tql
        └── use-foo.txt
```

## Cross-package usage

- `foo/tests/use-bar.tql` calls `bar::double`.
- `bar/tests/use-foo.tql` calls `foo::increment`.

Because `--package-dirs` points at the library, both packages are visible to
every test without additional flags.

You can also pin package discovery in a directory `test.yaml` by setting
`package-dirs:` there. This example keeps the entries commented so the CLI
flag remains the primary, explicit mechanism.
