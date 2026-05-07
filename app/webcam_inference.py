"""Live webcam anomaly detection with OpenCV.

Inference is heavy on CPU (~2-5s/frame for ViT-L/14 @ 518). To keep the
preview window responsive we run inference in a background thread and only
overlay the latest available result.
"""
from __future__ import annotations

import argparse
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np

from app.inference import predict
from app.utils import (
    DEFAULT_THRESHOLD,
    FAIL_COLOR_BGR,
    PASS_COLOR_BGR,
    apply_heatmap,
    setup_logging,
)

log = setup_logging()

WINDOW_TITLE = "Gear Defect Detection - Webcam (q to quit)"


def _draw_banner(frame_bgr: np.ndarray, score: float, verdict: str) -> None:
    color = PASS_COLOR_BGR if verdict == "PASS" else FAIL_COLOR_BGR
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.rectangle(frame_bgr, (12, 8), (120, 40), color, -1)
    cv2.putText(frame_bgr, verdict, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"score: {score:.3f}", (140, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _inference_worker(
    in_q: "Queue[np.ndarray | None]",
    out_q: "Queue[dict]",
    threshold: float,
) -> None:
    while True:
        frame_rgb = in_q.get()
        if frame_rgb is None:
            return
        try:
            result = predict(frame_rgb, threshold=threshold)
            out_q.put(result)
        except Exception:
            log.exception("Inference failed for a frame; skipping.")


def run(camera: int, threshold: float, width: int, height: int) -> int:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        log.error("No webcam detected at index %d. Exiting.", camera)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    in_q: Queue = Queue(maxsize=1)
    out_q: Queue = Queue(maxsize=1)
    worker = threading.Thread(
        target=_inference_worker, args=(in_q, out_q, threshold), daemon=True
    )
    worker.start()

    last_result: dict | None = None
    frame_count = 0
    fps_timer = time.time()
    log.info("Webcam started. Press 'q' to quit.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                log.warning("Failed to read frame; exiting.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if not in_q.full():
                in_q.put(frame_rgb)
            try:
                last_result = out_q.get_nowait()
            except Empty:
                pass

            display_bgr = frame_bgr.copy()
            if last_result is not None:
                overlay_rgb = apply_heatmap(
                    cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB),
                    last_result["anomaly_map"],
                    alpha=0.55,
                )
                display_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                _draw_banner(display_bgr, last_result["score"], last_result["verdict"])
            else:
                cv2.rectangle(display_bgr, (0, 0), (display_bgr.shape[1], 48), (0, 0, 0), -1)
                cv2.putText(display_bgr, "Warming up...", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_TITLE, display_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - fps_timer)
                fps_timer = now
                log.info("display ~%.1f fps", fps)
    finally:
        in_q.put(None)
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam anomaly detection.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"PASS/FAIL cutoff (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    return run(args.camera, args.threshold, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())
