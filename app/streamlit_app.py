"""Streamlit dashboard for gear defect detection."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Streamlit runs this file directly, so the project root isn't on sys.path
# the way it would be under `python -m app.streamlit_app`. Add it explicitly
# before any `from app.*` imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _cloud_env() -> str | None:
    """Detect a hosted Streamlit environment. Returns 'hf', 'streamlit', or None."""
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"):
        return "hf"
    if (
        "/mount/src/" in str(_PROJECT_ROOT)
        or os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
        or os.environ.get("HOSTNAME", "").startswith("streamlit")
    ):
        return "streamlit"
    return None


def _is_cloud() -> bool:
    return _cloud_env() is not None

import numpy as np
import streamlit as st
from PIL import Image

from app.inference import predict
from app.model import get_model
from app.utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_THRESHOLD,
    OUTPUTS_DIR,
    apply_heatmap,
    get_device,
    make_composite,
)

st.set_page_config(
    page_title="Gear Defect Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading AnomalyCLIP (first launch downloads ~890 MB CLIP weights — please wait)...")
def _warmup_model() -> str:
    loaded = get_model()
    return loaded.device


def _verdict_pill(verdict: str, score: float, threshold: float) -> None:
    color = "#1d8348" if verdict == "PASS" else "#c0392b"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:18px; padding:18px 22px;
                    border-radius:14px; background:{color}; color:white;
                    font-family:-apple-system,Segoe UI,Roboto,sans-serif;">
          <div style="font-size:34px; font-weight:700; letter-spacing:2px;">{verdict}</div>
          <div style="font-size:16px; opacity:0.9;">
            anomaly score <b>{score:.4f}</b> &nbsp;&middot;&nbsp; threshold {threshold:.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _composite_png_bytes(result: dict) -> bytes:
    composite = make_composite(
        result["original"], result["heatmap"], result["overlay"],
        result["score"], result["verdict"], result["threshold"],
    )
    buf = io.BytesIO()
    Image.fromarray(composite).save(buf, format="PNG")
    return buf.getvalue()


def _render_result(result: dict) -> None:
    _verdict_pill(result["verdict"], result["score"], result["threshold"])
    st.progress(min(max(result["score"], 0.0), 1.0), text=f"score = {result['score']:.4f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Original")
        st.image(result["original"], use_container_width=True)
    with col2:
        st.caption("Anomaly heatmap")
        st.image(result["heatmap"], use_container_width=True)
    with col3:
        st.caption("Overlay")
        st.image(result["overlay"], use_container_width=True)

    st.download_button(
        "Download composite PNG",
        data=_composite_png_bytes(result),
        file_name="gear_defect_result.png",
        mime="image/png",
    )


def main() -> None:
    st.title("Gear Defect Detection")
    st.caption("Zero-shot industrial anomaly detection powered by AnomalyCLIP.")

    cloud_env = _cloud_env()
    cloud = cloud_env is not None

    if not Path(DEFAULT_CHECKPOINT).exists():
        if cloud:
            st.error(
                "Model checkpoint not found at `models/anomalyclip.pth`.\n\n"
                "This file should have been committed to the repository. "
                "Verify it's present in your GitHub repo and redeploy."
            )
        else:
            st.error(
                "Model checkpoint not found.\n\n"
                "Run **`bash scripts/setup.sh`** in your terminal once to download "
                "the pretrained AnomalyCLIP weights, then refresh this page."
            )
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        threshold = st.slider(
            "PASS / FAIL threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
            help="Anomaly scores at or above this value are flagged FAIL.",
        )
        device = _warmup_model()
        device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
        st.markdown(f"**Device:** `{device_label}`")
        st.markdown(f"**Checkpoint:** `{DEFAULT_CHECKPOINT.name}`")
        if cloud_env == "hf":
            st.markdown("**Environment:** `Hugging Face Spaces`")
        elif cloud_env == "streamlit":
            st.markdown("**Environment:** `Streamlit Cloud`")
        with st.expander("How it works"):
            st.markdown(
                "AnomalyCLIP learns generic *normal* and *abnormal* text "
                "prompts on auxiliary data (MVTec AD), then transfers "
                "zero-shot to any new object class — no gear samples or "
                "retraining required.\n\n"
                "Outputs: per-pixel heatmap + image-level anomaly score."
            )
        st.divider()
        if not cloud:
            st.markdown(
                "**Continuous webcam mode** opens a native window:\n\n"
                "```bash\nbash scripts/webcam.sh\n```"
            )
            st.markdown(f"Outputs save to `{OUTPUTS_DIR.name}/`")
        else:
            st.markdown(
                "Continuous webcam mode (native OpenCV window) is "
                "**local-only** and unavailable on Streamlit Cloud. "
                "Use the **Webcam snapshot** tab for browser-side capture."
            )

    upload_tab, webcam_tab = st.tabs(["Upload image", "Webcam snapshot"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a gear image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input", width=320)
            if st.button("Run inference", type="primary", key="upload_btn"):
                with st.spinner("Running AnomalyCLIP..."):
                    result = predict(image, threshold=threshold)
                _render_result(result)

    with webcam_tab:
        st.markdown(
            "Take a snapshot using your browser's webcam — a one-shot "
            "alternative to the native OpenCV window."
        )
        snap = st.camera_input("Snapshot")
        if snap is not None:
            image = Image.open(snap).convert("RGB")
            with st.spinner("Running AnomalyCLIP..."):
                result = predict(image, threshold=threshold)
            _render_result(result)


if __name__ == "__main__":
    main()
