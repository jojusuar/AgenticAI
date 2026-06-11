#!/bin/sh

ls -al .
codegraph-mcp start &
codegraph-mcp watch /app/workspace &

exec python graph.py "$@"