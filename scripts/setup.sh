#!/usr/bin/env bash
# One-shot environment setup: install deps, vendor AnomalyCLIP, copy checkpoint, warm CLIP cache.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] uv sync"
uv sync

echo "[2/4] vendor AnomalyCLIP"
if [ ! -d "vendor/AnomalyCLIP" ]; then
    mkdir -p vendor
    git clone --depth 1 https://github.com/zqhang/AnomalyCLIP vendor/AnomalyCLIP
    rm -rf vendor/AnomalyCLIP/.git
    echo "    cloned vendor/AnomalyCLIP"
else
    echo "    already present, skipping clone"
fi

echo "[3/4] patch hardcoded cache_dir + copy pretrained checkpoint"
uv run python - <<'PY'
from pathlib import Path
import shutil

ml = Path("vendor/AnomalyCLIP/AnomalyCLIP_lib/model_load.py")
src = ml.read_text()
bad_path = "/remote-home/iot_zhouqihang/root/.cache/clip"
good_path = "~/.cache/clip"
if bad_path in src:
    ml.write_text(src.replace(bad_path, good_path))
    print(f"    [OK] patched {src.count(bad_path)} hardcoded path occurrence(s)")
else:
    print("    [skip] cache_dir patch not needed")

ckpt_src = Path("vendor/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth")
ckpt_dst = Path("models/anomalyclip.pth")
ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
if not ckpt_dst.exists():
    if not ckpt_src.exists():
        raise SystemExit(f"[FAIL] expected pretrained checkpoint at {ckpt_src}")
    shutil.copy2(ckpt_src, ckpt_dst)
    print(f"    [OK] copied {ckpt_src.name} -> {ckpt_dst}")
else:
    print(f"    [skip] {ckpt_dst} already present")
PY

echo "[4/4] warm CLIP cache (first run downloads ~890MB ViT-L/14@336px)"
uv run python -c "from app.model import get_model; get_model(); print('    [OK] model warmed')"

echo
echo "Setup complete. Next steps:"
echo "  uv run streamlit run app/streamlit_app.py    # dashboard"
echo "  bash scripts/webcam.sh                       # live webcam"
echo "  bash scripts/infer.sh path/to/image.jpg      # single image"
