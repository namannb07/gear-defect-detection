"""AnomalyCLIP loader: cached, zero-shot, ready to predict.

Patches sys.path so the vendored AnomalyCLIP code at vendor/AnomalyCLIP
can be imported as a sibling package.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.utils import (
    ANOMALYCLIP_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_SIZE,
    get_device,
    setup_logging,
)

log = setup_logging()

# Architecture knobs that match the shipped checkpoint at
# vendor/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth
PROMPT_LENGTH = 12
TEXT_EMBEDDING_DEPTH = 9
TEXT_EMBEDDING_LENGTH = 4
DPAM_LAYER = 20
FEATURE_LAYERS = [6, 12, 18, 24]
FEATURE_MAP_LAYER = [0, 1, 2, 3]


def _ensure_vendor_on_path() -> None:
    if not ANOMALYCLIP_DIR.exists():
        raise FileNotFoundError(
            f"AnomalyCLIP source not found at {ANOMALYCLIP_DIR}. "
            "Verify vendor/AnomalyCLIP is committed to the repo."
        )
    p = str(ANOMALYCLIP_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _patch_clip_cache_dir() -> None:
    """Defense-in-depth: force a sane CLIP cache dir even if a bad one is passed in."""
    import AnomalyCLIP_lib.model_load as ml

    safe = os.path.expanduser("~/.cache/clip")
    original = ml._download

    def patched(url: str, cache_dir: str | None = None):
        if not cache_dir or "/remote-home" in cache_dir or not os.access(
            os.path.dirname(cache_dir.rstrip("/")) or "/", os.W_OK
        ):
            cache_dir = safe
        return original(url, cache_dir)

    ml._download = patched


@dataclass
class LoadedModel:
    model: Any
    text_features: torch.Tensor
    preprocess: Any
    device: str
    image_size: int


_CACHE: dict[str, LoadedModel] = {}


def get_model(
    checkpoint_path: Path | str | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> LoadedModel:
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    cache_key = f"{ckpt}::{image_size}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Verify models/anomalyclip.pth is committed to the repo."
        )

    _ensure_vendor_on_path()
    _patch_clip_cache_dir()

    import AnomalyCLIP_lib  # noqa: E402
    from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD  # noqa: E402
    from AnomalyCLIP_lib.transform import image_transform  # noqa: E402
    from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402

    device = get_device()
    log.info("Loading AnomalyCLIP on device=%s", device)

    design = {
        "Prompt_length": PROMPT_LENGTH,
        "learnabel_text_embedding_depth": TEXT_EMBEDDING_DEPTH,
        "learnabel_text_embedding_length": TEXT_EMBEDDING_LENGTH,
    }
    log.info("Downloading/loading CLIP ViT-L/14@336px (first run pulls ~890MB into ~/.cache/clip)")
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=design)
    model.eval()

    log.info("Loading prompt learner from %s", ckpt)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    prompt_learner.load_state_dict(state["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)

    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    preprocess = image_transform(image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

    loaded = LoadedModel(
        model=model,
        text_features=text_features,
        preprocess=preprocess,
        device=device,
        image_size=image_size,
    )
    _CACHE[cache_key] = loaded
    log.info("Model ready (image_size=%d, device=%s)", image_size, device)
    return loaded
