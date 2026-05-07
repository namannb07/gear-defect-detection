#!/usr/bin/env bash
# AnomalyCLIP is zero-shot — no training required for inference.
# This script is a placeholder that points you to the upstream training entrypoint
# if you want to fine-tune the prompt embeddings on your own auxiliary dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

cat <<'EOF'
AnomalyCLIP is zero-shot. The shipped checkpoint at models/anomalyclip.pth was
trained once on MVTec AD and generalizes to new object classes (including gears)
without any retraining.

If you DO want to fine-tune the prompt embeddings on your own auxiliary
dataset, see the upstream training script at:

    vendor/AnomalyCLIP/train.sh
    vendor/AnomalyCLIP/train.py

This requires a labeled industrial anomaly dataset (e.g. MVTec AD, VisA) with
ground-truth masks. For simply running inference on gear images, no action is
needed — just use:

    uv run streamlit run app/streamlit_app.py
EOF
