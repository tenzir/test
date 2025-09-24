I would like to plan a new feature that supports integration testing in the
Tenzir Library, hosted at https://github.com/tenzir/library. 

The **library** is nothing but a collection of **packages**, each of which
corresponds to one (top-level) directory in the repository. To date, the package
format is simple: it just contains a package.yaml file, along with a package.svg
icon that's referenced inside package.yaml. Clone the repository locally so that
you can understand the structure of the library yourself.

We're in the process of changing the package structure substantially. Today we
have:

    example-package
    |- package.svg
    +- package.yaml

Moving forward, we want this directory layout:

    example-package
    |- operators
    |  |- foo
    |  |  +- bar.tql
    |  |  +- baz.tql
    |  |- qux
    |  |  +- corge.tql
    |- pipelines
    |  |  +- a-deployable-thing.tql
    |- tests
    +- package.yaml
    +- package.svg

This is the result of the idea that we're changing TQL so that it supports
*user-defined operators (UDOs)*. There's a natural mapping of a *.tql file to an
UDO:

- example-package/operators/foo/bar.tql → example::foo::bar
- example-package/operators/foo/baz.tql → example::foo:baz
- example-package/operators/qux/corge.tql → example::qux::corge

Paths under `example-package/` map to the `example` namespace as defined in its `package.yaml`.

These UDOs can be called from TQL just like regular operators, e.g.:

```tql
from {x: 42, y: 43}
where x > 42
example::foo::bar
example::foo::baz
select result
example::qux::corge
```

We're also doing this so that we can *test* UDOs in packages. This is what this
is all about. The idea is that the `tests` directory in the package is the
"project root" of our test harness. 

Now, we may want to register a new test runner for packages that sets the option
`--package-dirs`, a vector of file system paths where to search for packages.
The runner's job would be to set this option to the current package, i.e., the
parent directory of "tests" (called "example-package" in the above snippet), as part of
the invocation of the `tenzir` binary.

Here's the ultimate UX I experience. Say I have this package:

    sample
    |- operators
    |  |- foo
    |  |  +- bar.tql
    |- tests
    |  |- inputs
    |  |  +- input.json
    |  |- foo
    |  |  +- bar.tql
    |  |  +- bar.txt
    +- package.yaml
    +- package.svg

Note that I mimicked the operator directory hierarchy in the tests. (This could 
be a best practice.) I'd like to write tests/foo/bar.tql as follows:

```tql
from_file f"{env("TENZIR_INPUTS")}/input.json"
sample::foo::bar
```

For this to work, I *don't* want to user the runner abstraction that we have.
Rather, I'd like to have a configuration option/setting in `tenzir-test` for
package auto-discovery. This option would walk up from $CWD until it finds a
package.yaml that is describing a Tenzir package (check for keys `name` and
`description` initially)
