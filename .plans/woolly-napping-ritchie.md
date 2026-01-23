# Plan: Fix uvx fallback for tenzir-node

## Problem

The current uvx fallback for `tenzir-node` runs `uvx tenzir-node`, but `tenzir-node` is not a standalone package—it's a script within the `tenzir` package. The correct command is:

```bash
uvx --from tenzir tenzir-node
```

## Solution

Modify `_resolve_binary()` in `src/tenzir_test/config.py` to use `--from tenzir` syntax for `tenzir-node`:

**Current code (line 51-52):**

```python
if shutil.which("uvx"):
    return ("uvx", binary_name)
```

**Updated code:**

```python
if shutil.which("uvx"):
    if binary_name == "tenzir-node":
        return ("uvx", "--from", "tenzir", "tenzir-node")
    return ("uvx", binary_name)
```

## Files to Modify

1. **`src/tenzir_test/config.py`** (line 51-52)
   - Update `_resolve_binary()` to use `--from tenzir` for `tenzir-node`

2. **`tests/test_config.py`** (line 64)
   - Change expected value from `("uvx", "tenzir-node")` to `("uvx", "--from", "tenzir", "tenzir-node")`

3. **`.docs/src/content/docs/reference/test-framework/index.mdx`** (line 568)
   - Change `uvx tenzir-node` to `uvx --from tenzir tenzir-node`

4. **`.docs/src/content/docs/guides/testing/write-tests.mdx`** (line 18)
   - Change `uvx tenzir-node` to `uvx --from tenzir tenzir-node`

## Verification

1. Run unit tests: `uv run pytest tests/test_config.py -v`
2. Test manually with uvx available but no tenzir-node in PATH

## Commit & Push

After verification, commit and push changes in both repositories:

1. **Main repo** (`$PWD`): Commit code and test changes
2. **Docs repo** (`$PWD/.docs`): Commit documentation changes
