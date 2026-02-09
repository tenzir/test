This release fixes a serialization bug where Python fixture tests with `skip` configurations in `test.yaml` failed with a JSON serialization error.

## 🐞 Bug fixes

### Dataclass serialization in Python fixture configurations

Python fixture tests that use `skip` in their `test.yaml` no longer fail with `Object of type SkipConfig is not JSON serializable`.

*By @mavam and @claude.*
