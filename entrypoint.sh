#!/bin/sh

ls -al .

exec python graph.py "$@"
