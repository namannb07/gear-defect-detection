#!/usr/bin/env bash
# Run inference on a single image. Usage: bash scripts/infer.sh path/to/image.jpg
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/infer.sh <image_path> [--threshold 0.5] [--out path.png]"
    exit 1
fi

IMAGE="$1"
shift
uv run python -m app.inference --image "$IMAGE" "$@"
