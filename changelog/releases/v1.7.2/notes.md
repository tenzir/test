This release fixes enum serialization errors when Python fixture tests use configuration values like mode: sequential in test.yaml.

## 🐞 Bug fixes

### Enum serialization in Python fixture config

Python fixture tests could fail with serialization errors when the test configuration included enum values like `mode: sequential` in `test.yaml`. These values are now properly converted to strings before being passed to test scripts.

*By @mavam and @codex in #32.*
