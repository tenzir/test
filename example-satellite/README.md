# Example Satellite Project

This companion project demonstrates how a satellite can reuse runners and
fixtures from the root `example-project` while adding its own customizations.

## Layout

```
example-satellite/
├── fixtures.py
└── tests/
    ├── hex/reuse.{xxd,txt}
    └── satellite/marker.{sh,txt}
```

- `fixtures.py` registers the `satellite_marker` fixture that the shell test
  consumes.
- `tests/hex/reuse.xxd` exercises the `xxd` runner exported by the root project;
  no additional runners are needed in the satellite.
- `tests/satellite/marker.sh` verifies that the local fixture is loaded when the
  satellite participates in a multi-project run.

Run the example together with the main project:

```sh
uvx tenzir-test --root example-project example-satellite
```
