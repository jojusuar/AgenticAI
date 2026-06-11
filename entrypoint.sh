#!/bin/sh

ls -al .
codegraph-mcp start &
codegraph-mcp watch /app/workspace &
sleep 5

exec python graph.py "$@"