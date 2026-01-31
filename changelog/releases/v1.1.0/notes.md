This release adds stdin file support for piping data directly to tests, and improves satellite project display in project listings.

## 🚀 Features

### Stdin file support for piping data to tests

The test harness now supports `.stdin` files for piping data directly to test processes.

Place a `.stdin` file next to any test to have its contents automatically piped to the subprocess stdin. The harness also sets the `TENZIR_STDIN` environment variable with the file path. This is particularly useful for TQL tests where you can start a pipeline with a parser directly:

```tql
read_csv
where count > 10
```

Instead of using `.input` files with `from_file env("TENZIR_INPUT")`. Both approaches remain valid—choose whichever fits your test better.

Shell and Python tests can also read from stdin or access `TENZIR_STDIN` when needed. Custom runners can use `get_stdin_content(env)` to read the file contents and pass them via the `stdin_data` parameter to `run_subprocess()`.

*By @mavam and @claude.*

## 🔧 Changes

### Satellite projects show relative paths in project listing

When listing projects in the execution plan, satellite projects now display their path relative to the root project instead of just their directory name. This makes satellite projects with identical directory names distinguishable. Additionally, project markers have been refined: root projects use a filled marker (● for packages, ■ for regular projects), while satellite projects use an empty marker (○ for packages, □ for regular projects).

*By @mavam and @claude.*
