"""Single-image AnomalyCLIP inference.

Reproduces the math from vendor/AnomalyCLIP/test_one_example.py and wraps it
into a clean predict() callable returning score + heatmap + verdict.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from app.model import FEATURE_LAYERS, FEATURE_MAP_LAYER, DPAM_LAYER, get_model
from app.utils import (
    DEFAULT_THRESHOLD,
    OUTPUTS_DIR,
    apply_heatmap,
    make_composite,
    save_rgb,
    setup_logging,
    verdict_from_score,
)

log = setup_logging()

GAUSSIAN_SIGMA = 4
SOFTMAX_TEMP = 0.07


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(image.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def predict(
    image: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    checkpoint_path: Path | str | None = None,
) -> dict:
    loaded = get_model(checkpoint_path=checkpoint_path)
    pil = _to_pil(image)
    img_size = loaded.image_size

    original_resized = pil.resize((img_size, img_size), Image.BILINEAR)
    original_rgb = np.array(original_resized)

    tensor = loaded.preprocess(pil).unsqueeze(0).to(loaded.device)

    import AnomalyCLIP_lib  # type: ignore  # added to sys.path by app.model

    with torch.no_grad():
        image_features, patch_features = loaded.model.encode_image(
            tensor, FEATURE_LAYERS, DPAM_layer=DPAM_LAYER
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features @ loaded.text_features.permute(0, 2, 1)
        text_probs = (text_probs / SOFTMAX_TEMP).softmax(-1)
        score = float(text_probs[0, 0, 1].detach().cpu())

        anomaly_maps = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= FEATURE_MAP_LAYER[0]:
                pf = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(pf, loaded.text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], img_size)
                amap = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_maps.append(amap)

        anomaly_map = torch.stack(anomaly_maps).sum(dim=0)
        anomaly_map_np = anomaly_map.detach().cpu().numpy()[0]
        anomaly_map_np = gaussian_filter(anomaly_map_np, sigma=GAUSSIAN_SIGMA)

    heatmap_only_rgb = apply_heatmap(np.zeros_like(original_rgb), anomaly_map_np, alpha=0.0)
    overlay_rgb = apply_heatmap(original_rgb, anomaly_map_np, alpha=0.5)
    verdict = verdict_from_score(score, threshold)

    return {
        "score": score,
        "verdict": verdict,
        "threshold": threshold,
        "anomaly_map": anomaly_map_np,
        "original": original_rgb,
        "heatmap": heatmap_only_rgb,
        "overlay": overlay_rgb,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run AnomalyCLIP inference on a single image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image (jpg/png).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"PASS/FAIL cutoff on the anomaly score (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output composite path (default: outputs/<imagename>_result.png).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Override checkpoint path (default: models/anomalyclip.pth).")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    result = predict(args.image, threshold=args.threshold, checkpoint_path=args.checkpoint)

    out_path = args.out or (OUTPUTS_DIR / f"{args.image.stem}_result.png")
    composite = make_composite(
        result["original"],
        result["heatmap"],
        result["overlay"],
        score=result["score"],
        verdict=result["verdict"],
        threshold=result["threshold"],
    )
    save_rgb(composite, out_path)

    print()
    print(f"  Image     : {args.image}")
    print(f"  Score     : {result['score']:.4f}  (threshold {args.threshold:.2f})")
    print(f"  Verdict   : {result['verdict']}")
    print(f"  Composite : {out_path}")
    print()


if __name__ == "__main__":
    _cli()
