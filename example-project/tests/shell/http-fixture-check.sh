#!/bin/sh
# fixtures: [http]
# timeout: 10

set -eu

if [ -z "${HTTP_FIXTURE_URL:-}" ]; then
  echo "HTTP fixture URL not provided" >&2
  exit 1
fi

response=$(curl -sS -X POST \
  -H "Content-Type: application/json" \
  --data '{"ok": true}' \
  "$HTTP_FIXTURE_URL")

if [ "$response" != '{"ok": true}' ]; then
  echo "unexpected echo response: $response" >&2
  exit 1
fi
