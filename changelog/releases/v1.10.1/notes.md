Project fixtures can now declare their Python package dependencies inline with PEP 723 metadata. tenzir-test installs those dependencies before loading fixtures, so regular test runs and fixture mode work for projects with self-contained fixture modules.

## 🐞 Bug fixes

### Project fixture inline dependencies

Project fixtures can now declare Python package dependencies inline with PEP 723 metadata:

```python
# /// script
# dependencies = ["boto3"]
# ///
```

`tenzir-test` installs these dependencies with `uv` before importing fixture modules, so fixtures used during regular test runs and `tenzir-test --fixture` can manage their own Python package requirements.

*By @mavam and @codex in #47.*
