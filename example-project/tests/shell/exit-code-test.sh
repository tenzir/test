#!/bin/sh

# This test intentionally fails to demonstrate the arrow output

echo "Running test..."
echo "About to fail with exit code 42"
echo "Simulated error message on stderr" >&2
exit 42
