# Example Library

This directory contains two sibling packages, `foo` and `bar`, to demonstrate
**library mode**. Running `uvx tenzir-test --root example-library` discovers
both packages and runs their tests while making each package available to the
other.

## Layout

```
example-library/
├── foo/
│   ├── package.yaml
│   ├── operators/
│   │   └── increment.tql
│   └── tests/
│       ├── use-bar.tql
│       └── use-bar.txt
└── bar/
    ├── package.yaml
    ├── operators/
    │   └── double.tql
    └── tests/
        ├── use-foo.tql
        └── use-foo.txt
```

## Cross-package usage

- `foo/tests/use-bar.tql` calls `bar::double`.
- `bar/tests/use-foo.tql` calls `foo::increment`.

Because the root is a library, `tenzir-test` passes the library root as
`--package-dirs`, so both packages are visible to every test without extra
flags.
