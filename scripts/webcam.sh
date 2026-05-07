#!/usr/bin/env bash
# Live webcam anomaly detection. Press 'q' in the window to quit.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m app.webcam_inference "$@"
