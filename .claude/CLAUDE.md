# Repository Architecture

This document describes the architecture of `tenzir-test`, a reusable test harness for Tenzir.

## Core Concepts

The framework discovers test scenarios, executes them through pluggable runners, captures output, and compares against baseline files.

**Test Execution Flow:**

1. **CLI** parses arguments and invokes the execution engine
2. **Config** discovers settings from environment and CLI flags
3. **Engine** orchestrates the lifecycle: discovers tests, registers plugins, spawns workers, runs scenarios, and reports results
4. **Runners** execute tests based on file type (TQL, shell, Python)
5. **Fixtures** provide managed resources (Tenzir nodes, HTTP servers) with automatic lifecycle handling

## Package Layout

- `src/tenzir_test/` contains the core package with CLI, config, execution engine, runners, and fixtures
- `tests/` mirrors the package structure for unit tests
- `example-*/` directories provide reference implementations for different use cases

## Extension Points

**Runners** implement test execution for different file types. The framework includes runners for TQL queries, shell scripts, Python tests, and external commands. Projects can register custom runners.

**Fixtures** provide shared resources across tests. The built-in node fixture manages Tenzir processes. Projects define custom fixtures in a `fixtures/` directory that get auto-registered on discovery.

## Test Discovery

The engine walks directory trees looking for test files matching registered runner patterns. Each test pairs with a `.txt` baseline file for output comparison. A `test.yaml` file in any directory configures suite-level settings like timeouts and required fixtures.

## Multi-Project Support

The framework supports satellite projects that inherit fixtures and runners from a parent project, and package directories containing multiple Tenzir packages with cross-dependencies.

## Documentation

Primary documentation lives at <https://docs.tenzir.com/reference/test>.
