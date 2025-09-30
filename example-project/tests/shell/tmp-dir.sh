#!/bin/sh
# timeout: 10

set -eu

printf '%s\n' 'scratch dir ready'
printf '%s\n' 'payload from shell runner' >"$TENZIR_TMP_DIR/message.txt"
cat "$TENZIR_TMP_DIR/message.txt"
