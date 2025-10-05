#!/bin/sh
# fixtures: [satellite_marker]
# timeout: 5

set -eu

printf '%s\n' "satellite fixture: ${SATELLITE_MARKER:-<missing>}"
