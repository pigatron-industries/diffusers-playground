#!/usr/bin/env bash
# Run the LLM Test Harness locally, loading environment from .env.
#
# Usage:
#   ./run_local.sh                 # use .env values
#   ./run_local.sh --port 9000     # extra flags are passed through to app_nicegui.py
set -euo pipefail

# Resolve the script's own directory so it works from anywhere.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Prefer the project virtualenv if present.
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

exec "$PYTHON" app_nicegui.py "$@"
