#!/bin/sh
set -eu

# use PORT env var if set, otherwise default to 5000
PORT=${PORT:-5000}

# run uvicorn as PID 1 (using exec replaces the shell) (needed for signal handling)
exec uvicorn --app-dir src/oraku-api main:app --host 0.0.0.0 --port "$PORT"
