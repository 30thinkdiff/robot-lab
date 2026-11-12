#!/usr/bin/env bash
set -euo pipefail
exec python3 -X utf8 "$(dirname "$0")/server.py" --open "$@"
