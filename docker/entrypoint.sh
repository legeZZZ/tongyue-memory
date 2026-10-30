#!/bin/sh
# Start as root so a fresh volume is writable, then drop to the mneme user.
set -eu
db="${MNEME_DB:-/data/mneme.db}"
dir=$(dirname "$db")
mkdir -p "$dir"
chown mneme:mneme "$dir"
exec runuser -u mneme -- python -m mneme
