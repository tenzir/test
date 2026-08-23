#!/bin/sh
# pre-compare: sort

# The shell runner honors pre-compare too. This script lists files in whatever
# order the filesystem returns them, so the baseline stays stable only because
# both sides are sorted before comparison.

set -eu

mkdir -p "$TENZIR_TMP_DIR/spool"
touch "$TENZIR_TMP_DIR/spool/beta.log" "$TENZIR_TMP_DIR/spool/alpha.log"
find "$TENZIR_TMP_DIR/spool" -name '*.log' -exec basename {} \;
