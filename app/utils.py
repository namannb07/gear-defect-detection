"""Shared helpers: paths, device pick, heatmap rendering, logging."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
VENDOR_DIR = PROJECT_ROOT / "vendor"
ANOMALYCLIP_DIR = VENDOR_DIR / "AnomalyCLIP"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_CHECKPOINT = MODELS_DIR / "anomalyclip.pth"
DEFAULT_IMAGE_SIZE = 518
DEFAULT_THRESHOLD = 0.5

PASS_COLOR_BGR = (40, 180, 60)
FAIL_COLOR_BGR = (40, 40, 220)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("gear-defect")


def normalize_map(arr: np.ndarray) -> np.ndarray:
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - a_min) / (a_max - a_min)).astype(np.float32)


def apply_heatmap(image_rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a JET-colored anomaly map over an RGB image. Returns RGB uint8."""
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    h, w = image_rgb.shape[:2]
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = normalize_map(anomaly_map)
    heat_u8 = (norm * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * image_rgb.astype(np.float32) + (1 - alpha) * heat_rgb.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def verdict_from_score(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    return "FAIL" if score >= threshold else "PASS"


def make_composite(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    score: float,
    verdict: str,
    threshold: float,
) -> np.ndarray:
    """Build a labeled side-by-side composite (RGB uint8)."""
    h = max(original_rgb.shape[0], heatmap_rgb.shape[0], overlay_rgb.shape[0])
    panels = []
    for img, label in (
        (original_rgb, "Original"),
        (heatmap_rgb, "Heatmap"),
        (overlay_rgb, "Overlay"),
    ):
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_LINEAR)
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)

    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    row = np.concatenate(
        [panels[0], sep, panels[1], sep, panels[2]], axis=1
    )

    footer_h = 56
    footer = np.full((footer_h, row.shape[1], 3), 245, dtype=np.uint8)
    pill_color_rgb = (60, 180, 40) if verdict == "PASS" else (220, 40, 40)
    cv2.rectangle(footer, (12, 12), (140, 44), pill_color_rgb, -1)
    cv2.putText(footer, verdict, (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    text = f"Anomaly score: {score:.4f}   threshold: {threshold:.2f}"
    cv2.putText(footer, text, (160, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([row, footer], axis=0)


def save_rgb(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
