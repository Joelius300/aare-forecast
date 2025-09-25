#!/bin/sh
set -euo pipefail

# use PORT env var if set, otherwise default to 5000
PORT=${PORT:-5000}
# just to document the default port
EXPOSE 5000

# run uvicorn as PID 1 (using exec replaces the shell) (needed for signal handling)
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"