"""
OpenCV rendering helpers.

draw_frame() is the single public entry point — it mutates the frame in-place
and returns it, so callers can chain: writer.write(draw_frame(...)).
"""

import cv2
import numpy as np
from typing import Dict, Tuple

_GREEN = (0, 200, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN = (0, 220, 255)


def _text(frame, txt: str, pos: Tuple[int, int], scale: float = 0.7,
          thickness: int = 2, color: Tuple = _WHITE) -> None:
    """Draw text with a black outline so it reads on any background."""
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def draw_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    fallen_pins: Dict[int, float],
    pin_centroids: Dict[int, Tuple[float, float]],
    fallen_count: int,
    total_frames: int,
) -> np.ndarray:
    """
    Render all overlays onto frame (mutates in-place).

    fallen_pins     : {track_id: fall_timestamp_seconds}
    pin_centroids   : {track_id: (cx, cy)} for every pin visible this frame
                      (caller should include last-known position for occluded fallen pins)
    total_frames    : used to decide when to show the final summary (last 2 s)
    """
    elapsed = frame_idx / fps
    summary_start = max(0, total_frames - int(2 * fps))

    # --- Fallen-pin markers ---
    for tid, ts in fallen_pins.items():
        if tid not in pin_centroids:
            continue
        cx, cy = int(pin_centroids[tid][0]), int(pin_centroids[tid][1])
        cv2.circle(frame, (cx, cy), 20, _GREEN, -1)
        cv2.circle(frame, (cx, cy), 20, _WHITE, 2)
        _text(frame, _fmt(ts), (cx - 22, cy - 26), scale=0.55, thickness=1)

    # --- Corner HUD ---
    _text(frame, f"Time: {_fmt(elapsed)}", (12, 32), scale=0.8, thickness=2)
    _text(frame, f"Pins down: {fallen_count}", (12, 62), scale=0.8, thickness=2)

    # --- Final summary overlay (last 2 seconds) ---
    if frame_idx >= summary_start:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        bw, bh = 400, 130
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        cv2.rectangle(overlay, (x1, y1), (x1 + bw, y1 + bh), _BLACK, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        mid_x = x1 + bw // 2
        _text(frame, "GAME OVER", (mid_x - 90, y1 + 38),
              scale=1.05, thickness=2, color=_CYAN)
        _text(frame, f"Total time : {_fmt(elapsed)}", (mid_x - 110, y1 + 75),
              scale=0.75, thickness=2)
        _text(frame, f"Pins knocked down : {fallen_count}", (mid_x - 130, y1 + 112),
              scale=0.75, thickness=2)

    return frame
