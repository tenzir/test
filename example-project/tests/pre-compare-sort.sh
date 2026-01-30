#!/usr/bin/env bash
# pre-compare: sort

# Demonstrate pre-compare transform for handling non-deterministic output.
# This test produces lines in random order, but the sort transform ensures
# comparison succeeds against a sorted baseline.

echo "zebra"
echo "alpha"
echo "charlie"
echo "bravo"
